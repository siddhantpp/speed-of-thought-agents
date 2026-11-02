"""One-time sign-in for the google backend of gcal_server.py: writes token.json for YOUR Google account.

This is Google's own installed-app OAuth flow (developers.google.com/workspace/calendar/api/quickstart/python),
with write access to events instead of read-only. Nothing here runs in the demo or the tests.

  1. Google Cloud console: create a project, enable "Google Calendar API", then Google Auth platform > Clients >
     Create client > Application type "Desktop app". Download the JSON and save it as credentials.json.
  2. .venv/bin/pip install google-auth-oauthlib        (optional dependency, only for this step)
  3. .venv/bin/python mcp_servers/gcal_auth.py credentials.json token.json
     A browser window asks you to allow access; token.json is written next to it.
  4. Start the server with GCAL_BACKEND=google GCAL_TOKEN_FILE=/absolute/path/token.json

Keep credentials.json and token.json private (they are your keys): never commit them.
If you change SCOPES, delete token.json and sign in again.
"""
import sys

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]   # read and write events, nothing else


def main(client_file="credentials.json", token_file="token.json"):
    from google_auth_oauthlib.flow import InstalledAppFlow   # imported here so the demo never needs it
    creds = InstalledAppFlow.from_client_secrets_file(client_file, SCOPES).run_local_server(port=0)
    with open(token_file, "w") as f:
        f.write(creds.to_json())       # token, refresh_token, client_id, client_secret, expiry, ...
    print(f"wrote {token_file}; now set GCAL_BACKEND=google GCAL_TOKEN_FILE=<absolute path to it>")


if __name__ == "__main__":
    main(*sys.argv[1:3])
