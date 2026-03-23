## RunPod SSH Bootstrap Note

Exact custom-image SSH startup command copied from RunPod docs on 2026-03-23:

```bash
bash -c 'apt update; \
DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y; \
mkdir -p ~/.ssh; \
cd ~/.ssh; \
chmod 700 ~/.ssh; \
echo "$PUBLIC_KEY" >> authorized_keys; \
chmod 700 authorized_keys; \
service ssh start; \
sleep infinity'
```

Important accompanying requirement from the same docs:

- TCP port `22` must be exposed on the pod/template.

Why this note exists:

- The current RunPod SSH bringup issue is specifically about custom images like
  `slimerl/slime:v0.2.3` not being an obviously SSH-ready boot substrate.
- We want the exact docs version preserved verbatim while we clean up the more
  honest lowering around templates and boot/runtime separation.
