"""Group orchestrators: fan work out to workers in parallel, retry, then review as a batch."""
from concurrent.futures import ThreadPoolExecutor


class Group:
    """A domain lead (mid-size model) with a team of workers (small models).

    The lead never sees the raw inputs of every item, only the workers' compact
    JSON results, which is what keeps every model's context small.
    """

    def __init__(self, name, lead, workers: dict, tracer=None, max_parallel=4):
        self.name, self.lead, self.workers, self.tracer = name, lead, workers, tracer
        self.max_parallel = max_parallel

    def fan_out(self, worker: str, jobs: list[dict], retries: int = 1, check=None):
        """Run `worker` over many jobs concurrently.

        Each job is {"label", "task", optional "images", optional "schema"}. `check(job, result)` may
        return an error string; the worker is then retried with that feedback.
        Jobs that still fail come back as {"_failed": reason} for escalation.
        """
        agent = self.workers[worker]

        def one(job):
            task, last_error, result = job["task"], None, None
            for attempt in range(retries + 1):
                try:
                    extra = {"schema": job["schema"]} if "schema" in job else {}
                    result = agent.run(task, images=job.get("images"), label=job["label"], **extra)
                    last_error = check(job, result) if check else None
                    if not last_error:
                        return result
                except Exception as e:  # malformed output, timeout, ...
                    last_error = f"{type(e).__name__}: {e}"
                if self.tracer:
                    self.tracer.note(f"{agent.card.name} retry {attempt + 1} on {job['label']}: {last_error}")
                task = job["task"] + f"\n\nYour previous answer had a problem: {last_error}\nFix it and answer again."
            return {"_failed": last_error, "_last": result}

        with ThreadPoolExecutor(self.max_parallel) as pool:
            return list(pool.map(one, jobs))

    def review(self, task: str, label: str = "review"):
        return self.lead.run(task, label=label)
