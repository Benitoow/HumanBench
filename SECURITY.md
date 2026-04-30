# Security Policy

HumanBench is open source, but the repository must stay free of secrets and accidental private data.

## Do not commit

- `.env` or any local override file containing API keys
- private keys, certificates, or tokens
- generated benchmark outputs that contain sensitive prompts or responses
- personal data, screenshots, or logs that were not meant for publication

## Recommended repository protections

- keep branch protection enabled on the default branch
- require pull requests and at least one review before merge
- enable secret scanning and push protection in GitHub
- review diffs carefully before merging generated files
- use `.env.example` as the only shared configuration template

## Reporting a security issue

If you discover a secret in the repository or a configuration mistake that may expose credentials, remove it immediately and rotate the affected key. For anything more serious, open a private security report rather than a public issue.
