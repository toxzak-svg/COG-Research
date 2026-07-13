# Security Policy

## Scope

This public research surface contains documentation and, after review, sanitized reference implementations. It must not contain production secrets, private memory stores, personal conversations, or credentials.

## Reporting a vulnerability

Please report security issues privately to the repository owner through GitHub's private vulnerability reporting feature when available. Do not open a public issue containing:

- API keys or access tokens;
- personal data;
- private conversation excerpts;
- database contents;
- exploitable deployment details;
- credentials or session material.

Include:

- the affected file or component;
- reproduction steps;
- expected and observed behavior;
- potential impact;
- a minimal proof of concept that does not expose third-party data.

## Sensitive-data exposure

If a public commit appears to contain private Starfire state, credentials, personal data, or database artifacts, treat it as an incident rather than a normal bug.

Recommended response:

1. stop distributing or cloning the affected revision;
2. revoke exposed credentials immediately;
3. remove the material from the visible tree;
4. purge it from Git history when necessary;
5. rotate dependent secrets;
6. document the incident without reproducing the sensitive material.

## Supported versions

Only the current public branch and tagged public releases are supported. Private Starfire development builds are outside the scope of this repository.

## Research safety boundary

The repository does not claim that Starfire is safe for unrestricted autonomy, self-modification, production decision-making, or high-stakes deployment. Experimental mechanisms should run against synthetic fixtures and bounded objectives unless a separate safety review establishes otherwise.
