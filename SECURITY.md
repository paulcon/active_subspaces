# Security policy

## Supported versions

Only the latest minor release line receives fixes, including security fixes. For example, once
1.1.0 is released, fixes go into 1.1.x and the 1.0 line receives none.

| Version | Supported |
|---|---|
| latest 1.x minor release | yes |
| older 1.x minor releases | no |
| 0.1 and the 2016 code (tags `legacy-master`, `legacy-dev`, `0.1`) | no |

## Reporting a vulnerability

Report vulnerabilities privately through GitHub's private vulnerability reporting: on the
repository's **Security** tab, choose **Report a vulnerability**
(<https://github.com/paulcon/active_subspaces/security/advisories/new>). Do not open a public
issue.

Include the affected version, a description of the problem and, if possible, a minimal example.
The maintainer will acknowledge the report, keep you informed, and credit you in the advisory
unless you prefer otherwise.

## Scope

active_subspaces is a numerical library: it runs on your machine, with your data and your models.
Things to keep in mind:

- **Models are your code.** `evaluate` and the functions that take a model call the callables you
  pass, possibly in the executor you pass. The library does not sandbox them.
- **Loading results.** Results are saved as JSON metadata and NumPy arrays, never as pickles, and
  `load` checks their schema. Loading a file does not execute code from it.
- **No files or network.** The library writes no files unless you call a save function, and it
  makes no network connections.

## Secrets

The repository holds no secrets: no passwords, tokens or keys. Releases to PyPI use Trusted
Publishing, so no PyPI token is stored either.

The 2016 history of the `dev` branch contains a password for a Redis message broker, in
`celeryconfig.py`. The cluster and its Redis service no longer exist, so the credential grants
access to nothing. The git history was not rewritten. The Celery backend and its configuration are
not part of version 1.0.
