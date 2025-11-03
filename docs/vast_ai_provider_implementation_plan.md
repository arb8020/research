# Vast.ai Provider Implementation Plan

**Status**: Ready for implementation
**Target**: Get JAX basic integration test working with Vast.ai
**Date**: 2025-01-29

---

## Overview

This document provides a complete specification for implementing a Vast.ai provider for the broker system. The implementation will use **direct REST API calls** (no SDK dependency) following the same pattern as RunPod, PrimeIntellect, and Lambda Labs providers.

**📖 Before starting implementation, read Section 5 (Key Design Decisions)** - it answers common questions about:
- Where dependency installation happens (AFTER SSH, not during provisioning)
- When `wait_for_ssh_ready()` should return (when SSH works, not when scripts complete)
- SSH key management requirements (manual upload to Vast.ai console is required)
- Offer ID format and price handling

---

## 1. API Configuration

### Base URL
```python
VAST_API_BASE_URL = "https://console.vast.ai/api/v0"
```

### Authentication
- **Method**: API key in request headers
- **Header**: `Authorization: Bearer {api_key}`
- **Environment variable**: `VAST_API_KEY`
- **SSH keys**: Same as RunPod - user provides their own SSH key path

### API Key Setup
User must set: `export VAST_API_KEY=<your_vast_api_key_here>`

---

## 2. REST API Endpoints

### 2.1 Search for GPU Offers

**Endpoint**: `POST /api/v0/bundles/`

**Request Body**:
```json
{
  "verified": {"eq": true},
  "external": {"eq": false},
  "rentable": {"eq": true},
  "rented": {"eq": false},
  "order": [["dph_total", "asc"]],
  "type": "on-demand",
  "allocated_storage": 10
}
```

**Query Parameters** (optional filters):
- `num_gpus`: Number of GPUs (e.g., `{"eq": 1}`)
- `gpu_name`: GPU model filter (e.g., `{"contains": "RTX"}`)
- `dph_total`: Price per hour (e.g., `{"lte": 1.0}`)
- `cpu_ram`: RAM in MB (e.g., `{"gte": 16000}`)
- `disk_space`: Storage in GB (e.g., `{"gte": 100}`)
- `reliability2`: Reliability score 0-1 (e.g., `{"gte": 0.95}`)

**Response Format**:
```json
{
  "offers": [
    {
      "id": 12345678,
      "machine_id": 987654,
      "num_gpus": 1,
      "gpu_name": "RTX 3090",
      "gpu_ram": 24000,
      "cpu_cores_effective": 8,
      "cpu_ram": 32000,
      "disk_space": 100,
      "dph_total": 0.45,
      "inet_up": 100,
      "inet_down": 100,
      "reliability2": 0.98,
      "verification": "verified",
      "geolocation": "US",
      "cuda_max_good": 12.0,
      "driver_version": "535.104.05"
    }
  ]
}
```

### 2.2 Create Instance

**Endpoint**: `PUT /api/v0/asks/{offer_id}/`

**Request Body**:
```json
{
  "client_id": "me",
  "image": "runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204",
  "env": {},
  "price": 1.0,
  "disk": 50,
  "label": "jax-integration-test",
  "onstart": "echo 'Hello from startup script' >> /root/startup.log",
  "runtype": "ssh",
  "image_login": "",
  "python_utf8": false,
  "lang_utf8": false,
  "use_jupyter_lab": false,
  "jupyter_dir": "/",
  "force": false,
  "cancel_unavail": false
}
```

**Key Fields**:
- `client_id`: Always "me" (indicates current user)
- `image`: Docker image from Docker Hub
- `onstart`: **Shell script content to run on startup** (maps to `ssh_startup_script` parameter)
  - Runs after container starts
  - Useful for installing dependencies, configuring environment
  - Empty string `""` if no startup script needed
- `runtype`: Must be `"ssh"` for SSH access (alternatives: `"jupyter"`, `"jupyter-direct"`, `"ssh jupyter"`)
- `disk`: Container disk size in GB
- `label`: Instance name/label
- `price`: Maximum bid price (for spot instances) or accepted price (for on-demand)

**Response Format**:
```json
{
  "success": true,
  "new_contract": 789456,
  "msg": "creating instance..."
}
```

**Notes**:
- `runtype`: Must be "ssh" for SSH access (alternatives: "jupyter", "jupyter-direc", "ssh jupyter")
- The response returns `new_contract` which is the instance ID
- Instance creation is async - instance won't be ready immediately

### 2.3 Get Instance Details

**Endpoint**: `GET /api/v0/instances/?owner=me`

**Response Format**:
```json
{
  "instances": [
    {
      "id": 789456,
      "machine_id": 987654,
      "actual_status": "running",
      "num_gpus": 1,
      "gpu_name": "RTX 3090",
      "ssh_host": "ssh4.vast.ai",
      "ssh_port": 12345,
      "dph_total": 0.45,
      "image_uuid": "runpod/pytorch...",
      "reliability2": 0.98,
      "label": "jax-integration-test",
      "start_date": 1706553600.0,
      "duration": 1234.5,
      "extra_env": []
    }
  ]
}
```

**Status Values**:
- `"running"` - Instance is fully loaded and SSH accessible
- `"loading"` - Instance is starting up
- `"exited"` - Instance has stopped
- `"offline"` - Machine is offline

### 2.4 Terminate Instance

**Endpoint**: `DELETE /api/v0/instances/{instance_id}/`

**Response Format**:
```json
{
  "success": true,
  "msg": "destroying instance {instance_id}."
}
```

---

## 3. Data Type Mappings

### 3.1 GPUOffer Fields

| Vast.ai Field       | Broker Field       | Notes                                      |
|---------------------|--------------------|--------------------------------------------|
| `id`                | `id`               | Offer ID (used for provisioning)           |
| `num_gpus`          | `gpu_count`        | Number of GPUs                             |
| `gpu_name`          | `gpu_type`         | GPU model name                             |
| `gpu_ram`           | `vram_gb`          | GPU VRAM in MB (divide by 1000 for GB)     |
| `cpu_cores_effective` | `vcpu`           | Effective CPU cores                        |
| `cpu_ram`           | `memory_gb`        | RAM in MB (divide by 1000 for GB)          |
| `disk_space`        | `storage_gb`       | Disk space in GB                           |
| `dph_total`         | `price_per_hour`   | Total price per hour (USD)                 |
| `geolocation`       | `availability_zone`| Country code                               |
| `verification`      | -                  | "verified" or "unverified"                 |
| `reliability2`      | -                  | Reliability score 0-1                      |

**Cloud Type**:
- Vast.ai uses `"type": "on-demand"` (secure) or `"type": "bid"` (spot/community)
- Map `"on-demand"` → `CloudType.SECURE`
- Map `"bid"` → `CloudType.COMMUNITY`

### 3.2 GPUInstance Fields

| Vast.ai Field       | Broker Field       | Notes                                      |
|---------------------|--------------------|--------------------------------------------|
| `id`                | `id`               | Instance ID                                |
| `actual_status`     | `status`           | See status mapping below                   |
| `num_gpus`          | `gpu_count`        | Number of GPUs                             |
| `gpu_name`          | `gpu_type`         | GPU model name                             |
| `ssh_host`          | `public_ip`        | SSH hostname (e.g., "ssh4.vast.ai")        |
| `ssh_port`          | `ssh_port`         | SSH port number                            |
| `ssh_username`      | `ssh_username`     | **ALWAYS "root"** for Vast.ai              |
| `dph_total`         | `price_per_hour`   | Total price per hour                       |
| `label`             | `name`             | Instance label/name                        |

**Status Mapping**:
```python
{
    "running": InstanceStatus.RUNNING,
    "loading": InstanceStatus.PENDING,
    "exited": InstanceStatus.STOPPED,
    "offline": InstanceStatus.FAILED
}
```

---

## 4. Implementation Checklist

### 4.1 Core Functions (in `broker/broker/providers/vast.py`)

- [ ] `_make_api_request()` - HTTP helper with retry logic
- [ ] `search_gpu_offers()` - Search for available GPUs
- [ ] `provision_instance()` - Create instance from offer
- [ ] `get_instance_details()` - Get single instance by ID
- [ ] `list_instances()` - List all user instances
- [ ] `terminate_instance()` - Destroy instance
- [ ] `wait_for_ssh_ready()` - Poll until SSH accessible
- [ ] `get_fresh_instance()` - Alias for refresh (ProviderProtocol requirement)

### 4.2 Integration Points

**In `broker/broker/api.py`**:
- [ ] Add to `PROVIDER_MODULES` dict:
```python
PROVIDER_MODULES: dict[str, ProviderModule] = {
    "runpod": cast(ProviderModule, runpod),
    "primeintellect": cast(ProviderModule, primeintellect),
    "lambdalabs": cast(ProviderModule, lambdalabs),
    "vast": cast(ProviderModule, vast),  # ADD THIS
}
```

**In `broker/broker/providers/__init__.py`**:
- [ ] Add case to `get_provider_impl()`:
```python
elif provider_name == "vast":
    from . import vast
    return vast
```

**In `dev/jax_basic/run_integration_test.py`**:
- [ ] Add Vast.ai credentials:
```python
def get_credentials(provider_filter=None):
    credentials = {}
    if runpod_key := get_runpod_key():
        credentials["runpod"] = runpod_key
    if prime_key := get_prime_key():
        credentials["primeintellect"] = prime_key
    if lambda_key := get_lambda_key():
        credentials["lambdalabs"] = lambda_key
    if vast_key := os.getenv("VAST_API_KEY"):  # ADD THIS
        credentials["vast"] = vast_key          # ADD THIS
```

- [ ] Update `--provider` choices:
```python
parser.add_argument("--provider", type=str,
                    choices=["runpod", "primeintellect", "lambdalabs", "vast"],
                    help="Cloud provider to use")
```

**In `shared/config.py`** (if it exists):
- [ ] Add `get_vast_key()` helper function

---

## 5. Key Design Decisions

### 5.0 Design Philosophy Summary

This section clarifies common questions that may arise during implementation. The design follows patterns established by existing providers (RunPod, PrimeIntellect, Lambda Labs).

**Q: Where should dependency installation happen?**
- **A**: AFTER SSH is ready, NOT during provisioning. The JAX integration test installs dependencies via Bifrost after `wait_until_ssh_ready()` completes. The `onstart` parameter is available but NOT used by the integration test.

**Q: When should `wait_for_ssh_ready()` return?**
- **A**: When SSH connectivity is confirmed (test with `echo 'ssh_ready'`), NOT when startup scripts complete. See Section 12 Q2 for detailed explanation.

**Q: What format should offer IDs use?**
- **A**: `vast-{offer_id}` where `offer_id` is the numeric ID from Vast.ai's API. Example: `vast-12345678`. This follows the pattern used by RunPod (`runpod-...`) and Lambda Labs (`lambda-...`).

**Q: How should SSH keys be handled?**
- **A**: Users MUST manually upload their SSH public key to Vast.ai console before use (see Section 7.3). This is a PRE-REQUISITE. Vast.ai does not support programmatic SSH key upload.

**Q: What price should be sent in the provision request?**
- **A**: The exact `dph_total` from the offer (see Section 12 Q3). Store this in `GPUOffer.raw_data` during search, retrieve during provisioning.

**Q: Should the default container disk size (50GB) be changed?**
- **A**: No. 50GB matches RunPod's default and is appropriate for ML workloads. This is configurable via `container_disk_gb` parameter.

**Q: How does error recovery work when offers become unavailable?**
- **A**: The broker's `create()` method (in `broker/broker/api.py`) automatically retries with the next offer in the list. The integration test passes a list of 5 cheapest offers to `create()` with `n_offers=5`, enabling automatic fallback. Your `provision_instance()` function should return `None` when an offer is unavailable (operating error), not raise an exception. See Section 13 Q5 for details.

---

## 6. Implementation Notes

### 6.1 Price Normalization
- Vast.ai returns `dph_total` which is **total price for all GPUs**
- Must divide by `num_gpus` to get per-GPU price:
```python
price_per_gpu = offer["dph_total"] / offer["num_gpus"]
```

### 6.2 SSH Connection
- Vast.ai **ALWAYS** uses `root` as SSH username
- SSH hostname is typically `ssh[1-4].vast.ai`
- SSH port is a high random port (e.g., 12345-65535)
- Unlike RunPod, Vast.ai doesn't have proxy vs direct SSH - it's always direct

### 6.3 Instance Creation Flow
1. Call `POST /bundles/` to search for offers with filters:
   - `"verified": {"eq": True}` - Only verified hosts
   - `"reliability2": {"gte": min_reliability}` - Filter by reliability (default 0.95)
   - `"num_gpus": {"eq": gpu_count}` - Exact GPU count
   - `"order": [["dph_total", "asc"]]` - Sort by price
2. Post-filter offers in Python for additional criteria (GPU type, manufacturer, etc.)
3. Select an offer by `id` field from filtered results
4. Call `PUT /asks/{offer_id}/` with provisioning parameters:
   - `image`: Docker image (e.g., "runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204")
   - `onstart`: Startup script content (from `ssh_startup_script` parameter)
   - `disk`: Container disk size in GB
   - `runtype`: "ssh" for SSH access
5. Response returns `new_contract` which is the instance ID
6. Poll `GET /instances/?owner=me` until `actual_status == "running"`
7. Extract `ssh_host` and `ssh_port` from instance details (username is always "root")

### 6.4 Timeout Recommendations
- **Instance ready timeout**: 300 seconds (5 minutes)
- **SSH ready timeout**: 300 seconds (5 minutes)
- Vast.ai is generally faster than RunPod/Lambda Labs

### 6.5 Query Filters & GPU Type Handling

**Vast.ai Query Format** uses operator objects:
```python
# Examples:
{"eq": value}      # Equals
{"neq": value}     # Not equals
{"gt": value}      # Greater than
{"gte": value}     # Greater than or equal
{"lt": value}      # Less than
{"lte": value}     # Less than or equal
{"in": [v1, v2]}   # In list
{"contains": str}  # String contains (case-sensitive)
```

**Handling `gpu_type` from Broker Query Interface**:

The broker uses a **pandas-style query DSL** where users write:
```python
client.gpu_type.contains("H100")  # Query interface
```

This must be translated to Vast.ai's query format. Following **Tiger Style** (explicit contracts) and **Casey Muratori** (granularity + decoupling), the recommendation is:

**✅ Recommended Approach: Direct Translation with Assertions**

```python
def search_gpu_offers(...) -> List[GPUOffer]:
    """Search for available GPU offers on Vast.ai"""

    # Tiger Style: Assert preconditions on API inputs
    assert api_key is not None, "API key required"
    assert gpu_count > 0, f"gpu_count must be positive, got {gpu_count}"

    # Build base query with sensible defaults (explicit, not implicit)
    query = {
        "verified": {"eq": True},      # Only verified hosts
        "external": {"eq": False},     # Exclude external hosts
        "rentable": {"eq": True},      # Only rentable machines
        "rented": {"eq": False},       # Only available machines
        "order": [["dph_total", "asc"]],  # Sort by price ascending
        "type": "on-demand",           # Default to on-demand (not spot)
    }

    # Handle gpu_type parameter (if provided by broker)
    # Broker sends gpu_type as string from ProvisionRequest
    if manufacturer:
        # Vast.ai doesn't have direct manufacturer filter
        # Apply post-query filtering in Python (decoupled from API)
        pass

    # Casey Muratori: Provide BOTH high-level AND low-level access
    # High-level: Simple string match
    # Low-level: Allow custom query building

    # Tiger Style: Split GPU type filtering into two clear steps:
    # 1. Query Vast.ai API (broad search)
    # 2. Filter results in Python (precise control)

    response = _make_api_request("POST", "/bundles/", data=query, api_key=api_key)
    offers_raw = response.get("offers", [])

    # Post-process filtering (decoupled from API call)
    offers = []
    for offer_data in offers_raw:
        gpu_name = offer_data.get("gpu_name", "")

        # Tiger Style: Assert data validity from external API
        assert isinstance(gpu_name, str), f"gpu_name must be string, got {type(gpu_name)}"

        # Apply manufacturer filter (if specified)
        if manufacturer:
            mfr_lower = manufacturer.lower()

            # Vast.ai supports NVIDIA and AMD (as of May 2024)
            is_nvidia = any(x in gpu_name.lower() for x in ["rtx", "gtx", "quadro", "tesla", "a100", "h100", "v100", "a6000", "a5000", "l40", "l4"])
            is_amd = any(x in gpu_name.lower() for x in ["radeon", "instinct", "rx ", "mi100", "mi200", "mi300"])

            if mfr_lower == "nvidia" and not is_nvidia:
                continue
            elif mfr_lower == "amd" and not is_amd:
                continue
            elif mfr_lower not in ["nvidia", "amd"]:
                # Intel, other manufacturers not yet supported
                continue

        # Convert to GPUOffer...
        offers.append(GPUOffer(...))

    return offers
```

**Why This Approach?**

1. **Tiger Style Safety**:
   - Explicit precondition assertions at function entry
   - Explicit data validation from external API
   - No implicit assumptions about Vast.ai behavior

2. **Casey Muratori Decoupling**:
   - Separates API call from filtering logic
   - Provides "granularity" - user can search broadly, filter precisely
   - Avoids coupling to Vast.ai's query syntax limitations

3. **Vast.ai Limitation**:
   - Vast.ai's `gpu_name` filter uses `{"contains": "H100"}` but is **case-sensitive**
   - User might search `"h100"` vs `"H100"` - need case-insensitive matching
   - Better to do broad search + Python filter for flexibility

4. **Parameter Redundancy** (Casey Muratori):
   - Provide BOTH `gpu_type` parameter AND ability to pass raw query
   - Don't force users into one granularity level

**Alternative: Direct Query Translation** (simpler but less flexible):
```python
# If you want tight coupling to Vast.ai syntax:
if gpu_type:
    # Vast.ai expects gpu_name, not gpu_type
    query["gpu_name"] = {"contains": gpu_type}  # CASE-SENSITIVE!
```

❌ **Why this is worse**:
- Couples broker to Vast.ai's exact field names (`gpu_name` vs `gpu_type`)
- Case-sensitive matching breaks user expectations
- Integration discontinuity when user needs case-insensitive search

**Final Recommendation**: Use post-query Python filtering for `gpu_type` to maintain **decoupling** and **flexibility**. Only use direct Vast.ai query params for fields that have exact API support (like `num_gpus`, `dph_total`).

### 6.6 Error Handling

**HTTP Error Codes**:
- `401 Unauthorized` - Invalid API key (check `VAST_API_KEY`)
- `400 Bad Request` - Invalid parameters (e.g., offer already rented, invalid image name)
- `404 Not Found` - Instance/offer doesn't exist
- `429 Too Many Requests` - Rate limit exceeded (rare, but possible)
- `500/502/503` - Vast.ai server errors (should retry)

**Response Format**:
```python
# Success response
{"success": True, "new_contract": 12345}

# Error response
{"success": False, "msg": "Offer no longer available"}
```

**Error Handling Pattern** (following RunPod/Lambda Labs):

```python
@retry(max_attempts=3, delay=1, backoff=2, exceptions=(requests.RequestException, requests.Timeout))
def _make_api_request(method: str, endpoint: str, data: Optional[Dict] = None,
                     params: Optional[Dict] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Make a REST API request to Vast.ai with automatic retries.

    Tiger Style: Assert preconditions
    """
    if not api_key:
        raise ValueError("Vast.ai API key is required but was not provided")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    url = f"{VAST_API_BASE_URL}{endpoint}"
    logger.debug("Vast.ai API request %s %s (api key ...%s)",
                 method, url, api_key[-4:] if api_key else "none")

    try:
        response = requests.request(
            method=method,
            url=url,
            json=data,
            params=params,
            headers=headers,
            timeout=(10, 30),  # (connect_timeout, read_timeout)
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        logger.error("Vast.ai API request timed out")
        raise  # Will be retried by @retry decorator
    except requests.RequestException as exc:
        logger.error(f"Vast.ai API request failed: {exc}")
        raise  # Will be retried by @retry decorator

    # Handle empty responses (e.g., DELETE operations)
    if response.status_code == 204 or not response.content:
        return {}

    return response.json()
```

**Provider-Specific Error Handling**:

In `provision_instance()`:
```python
try:
    data = _make_api_request("PUT", f"/asks/{offer_id}/", data=request_body, api_key=api_key)

    # Tiger Style: Assert response validity
    assert "success" in data, f"Vast.ai response missing 'success' field: {data}"

    if not data["success"]:
        # Expected operating error (offer unavailable, etc.)
        logger.warning(f"Vast.ai provisioning failed: {data.get('msg', 'Unknown error')}")
        return None  # Signal failure to caller

    assert "new_contract" in data, f"Successful response missing 'new_contract': {data}"
    instance_id = data["new_contract"]

    # Tiger Style: Assert postcondition
    assert isinstance(instance_id, int), f"instance_id must be int, got {type(instance_id)}"

    return GPUInstance(...)

except ValueError as e:
    # Programmer error - invalid parameters
    logger.error(f"Invalid parameters for Vast.ai provisioning: {e}")
    raise  # Re-raise programmer errors

except Exception as e:
    # Unexpected error
    logger.error(f"Unexpected error provisioning Vast.ai instance: {e}", exc_info=True)
    return None  # Treat as operating error
```

**Key Points**:
1. Use `@retry` decorator with exponential backoff for network errors
2. Always check `response["success"]` field before processing
3. Return `None` for expected failures (offer unavailable) - don't raise
4. Raise `ValueError` for programmer errors (invalid params)
5. Log all errors with context for debugging

---

## 7. Testing Plan

### 7.1 Unit Testing (Manual)
```bash
# Set API key
export VAST_API_KEY=<your_vast_api_key_here>

# Test search
python -c "from broker import GPUClient; client = GPUClient({'vast': 'YOUR_KEY'}, '~/.ssh/id_ed25519'); print(client.search(client.gpu_type.contains('RTX')))"

# Test create (manual cleanup required!)
python -c "from broker import GPUClient; client = GPUClient({'vast': 'YOUR_KEY'}, '~/.ssh/id_ed25519'); offers = client.search(); instance = client.create(offers[0]); print(instance.id if instance else 'FAILED')"
```

### 7.2 Integration Testing
```bash
# Run JAX basic integration test
cd /Users/chiraagbalu/research
python dev/jax_basic/run_integration_test.py --provider vast
```

**Expected output**:
```
====================================================================
JAX GPU Integration Test
Provider: vast
====================================================================
✓ Instance created: 789456
✓ Instance is RUNNING
✓ SSH ready: ssh4.vast.ai:12345
✓ Code deployed to: /root/research
✓ Dependencies installed
✓ Test OUTPUT:
====================================================================
JAX GPU Integration Test
====================================================================
1. Importing JAX...
   ✅ JAX imported successfully
2. Checking GPU devices...
   ✅ Found 1 GPU device(s)
3. Running GPU computation...
   ✅ Matrix multiplication successful
4. Verifying GPU was used...
   ✅ Computation ran on GPU
====================================================================
🎉 All tests passed!
====================================================================
✓ Instance terminated
Cost estimate: ~$0.0075
====================================================================
INTEGRATION TEST PASSED
====================================================================
```

---

## 8. Known Limitations & Caveats

### 8.1 Spot Instances (Type "bid")
- Vast.ai's "bid" instances can be interrupted
- Not recommended for JAX tests - use "on-demand" only
- Filter by setting `"type": "on-demand"` in search query

### 8.2 Reliability Filtering
- **Always filter** by `reliability2 >= 0.95` to avoid unreliable hosts
- Default search includes `"verified": {"eq": true}` which helps

### 8.3 SSH Key Management (PRE-REQUISITE)

**⚠️ CRITICAL: Users MUST complete this setup before using Vast.ai provider**

Before using the Vast.ai provider, users MUST:

1. **Generate SSH key pair** (if not already done):
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
   ```

2. **Upload public key to Vast.ai console**:
   - Go to https://console.vast.ai/account/
   - Navigate to "SSH Keys" section
   - Add your public key content from `~/.ssh/id_ed25519.pub`
   - Vast.ai will automatically inject this key into all provisioned instances

3. **Provide private key path to broker**:
   ```python
   client = GPUClient(
       credentials={"vast": "YOUR_API_KEY"},
       ssh_key_path="~/.ssh/id_ed25519"  # Must match uploaded public key
   )
   ```

**How It Works**:
- Vast.ai API does NOT support programmatic SSH key upload (unlike RunPod/PrimeIntellect)
- All SSH keys in your Vast.ai account are automatically injected into instances
- Broker uses your private key for SSH connections after provisioning
- If keys don't match, SSH will fail with "Permission denied (publickey)"

**Troubleshooting**:
```bash
# Test SSH connection manually
ssh -i ~/.ssh/id_ed25519 -p <port> root@<host>

# If permission denied, verify:
1. Public key is in Vast.ai console
2. Private key path is correct
3. Private key permissions: chmod 600 ~/.ssh/id_ed25519
```

### 8.4 Docker Image Compatibility
- Not all Docker images work on Vast.ai
- Some hosts have restricted Docker registries
- Recommend using popular images (RunPod, NVIDIA, etc.)

### 8.5 Instance Startup Time
- "loading" status can take 1-5 minutes
- Some hosts are slower than others
- `reliability2` score correlates with startup speed

---

## 9. File Structure

```
broker/broker/providers/
├── __init__.py          # Add vast to get_provider_impl()
├── vast.py              # NEW FILE - Vast.ai provider implementation
├── runpod.py
├── primeintellect.py
└── lambdalabs.py

broker/broker/
├── api.py               # Add vast to PROVIDER_MODULES
├── client.py
└── types.py

dev/jax_basic/
└── run_integration_test.py  # Add vast credentials + provider choice

shared/
└── config.py            # Add get_vast_key() helper (optional)
```

---

## 10. Example Implementation Skeleton

```python
# broker/broker/providers/vast.py
"""
Vast.ai provider implementation
"""

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from ..types import CloudType, GPUInstance, GPUOffer, InstanceStatus, ProvisionRequest
from shared.retry import retry

logger = logging.getLogger(__name__)

VAST_API_BASE_URL = "https://console.vast.ai/api/v0"


@retry(max_attempts=3, delay=1, backoff=2, exceptions=(requests.RequestException, requests.Timeout))
def _make_api_request(method: str, endpoint: str, data: Optional[Dict] = None,
                     params: Optional[Dict] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Make a REST API request to Vast.ai with automatic retries."""
    if not api_key:
        raise ValueError("Vast.ai API key is required but was not provided")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    url = f"{VAST_API_BASE_URL}{endpoint}"
    logger.debug("Vast.ai API request %s %s (api key ...%s)", method, url, api_key[-4:] if api_key else "none")

    # TODO: Implement request logic
    pass


def search_gpu_offers(cuda_version: Optional[str] = None, manufacturer: Optional[str] = None,
                      memory_gb: Optional[int] = None, container_disk_gb: Optional[int] = None,
                      gpu_count: int = 1, min_reliability: float = 0.95,
                      api_key: Optional[str] = None) -> List[GPUOffer]:
    """Search for available GPU offers on Vast.ai

    Args:
        cuda_version: CUDA version requirement (not used by Vast.ai API directly)
        manufacturer: GPU manufacturer filter (Vast.ai is NVIDIA-only)
        memory_gb: Minimum system RAM in GB
        container_disk_gb: Minimum disk space in GB
        gpu_count: Number of GPUs required (default: 1)
        min_reliability: Minimum reliability score 0-1 (default: 0.95)
                        Set to 0 to disable reliability filtering
        api_key: Vast.ai API key

    Returns:
        List of GPUOffer objects sorted by price ascending

    Tiger Style: All parameters have explicit types and defaults
    """
    # TODO: Implement search logic
    pass


def provision_instance(request: ProvisionRequest, ssh_startup_script: Optional[str] = None,
                      api_key: Optional[str] = None) -> Optional[GPUInstance]:
    """Provision a GPU instance on Vast.ai

    Args:
        request: ProvisionRequest containing instance configuration
                 request.image: Docker image (default from ProvisionRequest is fine)
                               Vast.ai supports standard Docker Hub images
        ssh_startup_script: Shell script to run on startup (optional)
                           Maps to Vast.ai's "onstart" field
        api_key: Vast.ai API key

    Returns:
        GPUInstance if successful, None if provisioning failed

    Tiger Style: Returns None for operating errors (offer unavailable),
                raises ValueError for programmer errors (invalid params)

    Note on Docker images:
        - "runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204" (recommended default)
        - "nvidia/cuda:12.1.0-base-ubuntu22.04"
        - Any public Docker Hub image

    Note on startup scripts:
        - Vast.ai's "onstart" field accepts shell script content
        - Script runs after container starts (similar to RunPod/PrimeIntellect)
        - Example: "apt-get update && apt-get install -y vim"
        - Lambda Labs doesn't support startup scripts, but Vast.ai does

    Implementation:
        request_body = {
            "client_id": "me",
            "image": request.image,
            "onstart": ssh_startup_script or "",  # Use onstart field
            "runtype": "ssh",
            "disk": request.container_disk_gb or 50,
            # ... other fields
        }
    """
    # TODO: Implement provisioning logic
    pass


def get_instance_details(instance_id: str, api_key: Optional[str] = None) -> Optional[GPUInstance]:
    """Get details of a specific instance"""
    # TODO: Implement get instance logic
    pass


def list_instances(api_key: Optional[str] = None) -> List[GPUInstance]:
    """List all user's instances"""
    # TODO: Implement list instances logic
    pass


def terminate_instance(instance_id: str, api_key: Optional[str] = None) -> bool:
    """Terminate a Vast.ai instance"""
    # TODO: Implement termination logic
    pass


def wait_for_ssh_ready(instance, timeout: int = 300) -> bool:
    """Vast.ai-specific SSH waiting implementation"""
    # TODO: Implement SSH waiting logic
    pass


def get_fresh_instance(instance_id: str, api_key: str):
    """Alias for get_instance_details (ProviderProtocol requirement)"""
    return get_instance_details(instance_id, api_key=api_key)
```

---

## 11. Next Steps for Implementation Team

1. **Create** `broker/broker/providers/vast.py` with full implementation
2. **Update** `broker/broker/api.py` to register Vast.ai provider
3. **Update** `broker/broker/providers/__init__.py` to include Vast.ai
4. **Update** `dev/jax_basic/run_integration_test.py` for Vast.ai support
5. **Test** with API key from `.env` file (ensure `VAST_API_KEY` is set)
6. **Ensure** SSH public key is added to Vast.ai account before testing
7. **Run** JAX integration test: `python dev/jax_basic/run_integration_test.py --provider vast`

---

## 12. Success Criteria

✅ Search returns GPU offers sorted by price
✅ Instance creation succeeds and returns instance ID
✅ Instance status polling works until "running"
✅ SSH connection details are correctly extracted
✅ Bifrost can connect and run commands
✅ JAX GPU test passes on Vast.ai instance
✅ Instance termination succeeds
✅ Total cost < $0.10 for test run

---

## 13. Implementation FAQ

### Q1: SSH Key Configuration - How does the broker know which key to use?

**Answer**: The broker uses a **two-tier SSH key system** that matches the existing pattern in RunPod/PrimeIntellect:

1. **User must manually upload their SSH public key** to Vast.ai via console: https://console.vast.ai/account/
2. **Broker stores the matching private key path** via `GPUClient(ssh_key_path="~/.ssh/id_ed25519")`
3. **Private key is used for SSH connections** (via `instance.exec()` or Bifrost)

**Key Requirements**:
- The public key uploaded to Vast.ai must match the private key in `ssh_key_path`
- Vast.ai automatically injects all account SSH keys into provisioned instances
- Unlike RunPod (which supports API key upload), Vast.ai requires manual console upload

**Documentation Update Needed** (Section 7.3):
```markdown
### 7.3 SSH Key Management

**CRITICAL: Pre-requisite Setup**

Before using the Vast.ai provider, users MUST:

1. **Generate SSH key pair** (if not already done):
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
   ```

2. **Upload public key to Vast.ai**:
   - Go to https://console.vast.ai/account/
   - Navigate to "SSH Keys" section
   - Add your public key content from `~/.ssh/id_ed25519.pub`
   - Vast.ai will inject this key into all instances you provision

3. **Provide private key path to broker**:
   ```python
   client = GPUClient(
       credentials={"vast": "YOUR_API_KEY"},
       ssh_key_path="~/.ssh/id_ed25519"  # Must match uploaded public key
   )
   ```

**How It Works**:
- Vast.ai API does NOT support programmatic SSH key upload (unlike RunPod)
- All SSH keys in your Vast.ai account are automatically injected into instances
- Broker uses your private key for SSH connections after provisioning
- If keys don't match, SSH will fail with "Permission denied (publickey)"

**Troubleshooting**:
```bash
# Test SSH connection manually
ssh -i ~/.ssh/id_ed25519 -p <port> root@<host>

# If permission denied, verify:
1. Public key is in Vast.ai console
2. Private key path is correct
3. Private key permissions: chmod 600 ~/.ssh/id_ed25519
```
```

---

### Q2: Startup Script Timing - Does "running" mean the script is finished?

**Answer**: **No**. The `actual_status == "running"` means the **container is up**, NOT that `onstart` script completed.

**Timeline**:
1. `actual_status == "loading"` - Container is starting (1-5 min)
2. `actual_status == "running"` - Container is running, `onstart` script may still be executing
3. SSH becomes available ~10-30s after "running"
4. `onstart` script runs in background after SSH is ready

**IMPORTANT: How the JAX Integration Test Works**:

The JAX integration test **does NOT use startup scripts during provisioning**. Instead:

1. **Provisioning phase** (lines 60-65 of `run_integration_test.py`):
   ```python
   instance = gpu_client.create(
       offers,
       image="runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204",
       name="jax-integration-test",
       n_offers=len(offers)
   )
   # No ssh_startup_script parameter!
   ```

2. **Wait for SSH** (lines 84-88):
   ```python
   ssh_ready = instance.wait_until_ssh_ready(timeout=ssh_timeout)
   ```

3. **Install dependencies AFTER SSH is ready** (lines 93-102):
   ```python
   bootstrap_cmd = [
       """if ! command -v uv &> /dev/null; then
       curl -LsSf https://astral.sh/uv/install.sh | sh
       export PATH="$HOME/.cargo/bin:$PATH"
   fi""",
       "uv sync --extra dev-jax-basic"
   ]
   bifrost_client.push(bootstrap_cmd=bootstrap_cmd)
   ```

**Implementation Requirement for `wait_for_ssh_ready()`**:

The `wait_for_ssh_ready()` function should:
1. Poll until `actual_status == "running"`
2. Poll until SSH details (`ssh_host`, `ssh_port`) are populated
3. Test SSH connectivity with simple command (e.g., `echo 'ssh_ready'`)
4. **NOT wait for `onstart` completion** - dependency installation happens separately via Bifrost

**Why This Pattern?**
- Separates infrastructure provisioning from application setup
- User/test code controls dependency installation timing
- Consistent with RunPod/PrimeIntellect/Lambda Labs implementations:
  - **RunPod** (lines 797-916 of `runpod.py`): Tests SSH with `echo 'ssh_ready'`
  - **PrimeIntellect** (lines 365-458 of `primeintellect.py`): Tests SSH with `echo 'ssh_ready'`
  - **Lambda Labs** (lines 409-431 of `lambdalabs.py`): No SSH test (key not available at provider level)

**Reference Implementation** (from RunPod provider):
```python
def wait_for_ssh_ready(instance, timeout: int = 900) -> bool:
    # Wait for RUNNING status
    if not _wait_until_running(instance, timeout):
        return False

    # Wait for direct SSH assignment
    if not _wait_for_direct_ssh_assignment(instance, time.time(), timeout):
        return False

    # Test connectivity
    return _test_ssh_connectivity(instance)

def _test_ssh_connectivity(instance) -> bool:
    logger.info("Direct SSH ready! Waiting 30s for SSH daemon...")
    time.sleep(30)

    try:
        result = instance.exec("echo 'ssh_ready'", timeout=30)
        if result.success and "ssh_ready" in result.stdout:
            logger.info("SSH connectivity confirmed!")
            return True
        else:
            logger.warning(f"SSH test failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"SSH connection error: {e}")
        return False
```

---

### Q3: Price Parameter - Should it match `dph_total` or be higher?

**Answer**: **Set `price` to `offer["dph_total"]`** (exact match).

**IMPORTANT: Offer ID Format and Price Extraction**

The broker creates offer IDs in the format `vast-{offer_id}` where `offer_id` is the numeric ID from Vast.ai's API. For example:
- Vast.ai API returns: `{"id": 12345678, "dph_total": 0.45, ...}`
- Broker creates: `GPUOffer(id="vast-12345678", ...)`

**Implementation**:
```python
def provision_instance(request: ProvisionRequest, ssh_startup_script: Optional[str] = None,
                      api_key: Optional[str] = None) -> Optional[GPUInstance]:
    # Extract offer_id from request.gpu_type
    # Format: vast-{offer_id}
    if not request.gpu_type or not request.gpu_type.startswith("vast-"):
        logger.error(f"Invalid gpu_type format: {request.gpu_type}")
        return None

    offer_id = int(request.gpu_type.split("-")[1])  # Extract numeric ID

    # Get price from raw_data (stored in GPUOffer during search)
    # The GPUOffer.raw_data contains the original Vast.ai API response
    # RECOMMENDED: Store dph_total in GPUOffer during search, retrieve here

    # Vast.ai expects total price (not per-GPU), which matches dph_total from search
    price = offer_data["dph_total"]  # Use exact price from search

    request_body = {
        "client_id": "me",
        "image": request.image,
        "price": price,  # EXACT match to dph_total
        "disk": request.container_disk_gb or 50,
        "label": request.name or f"vast-{offer_id}-{int(time.time())}",
        "onstart": ssh_startup_script or "",
        "runtype": "ssh",
        # ... other fields
    }

    # Call Vast.ai API
    response = _make_api_request("PUT", f"/asks/{offer_id}/", data=request_body, api_key=api_key)
```

**Why exact match?**
- Vast.ai rejects if `price < dph_total` (offer unavailable error)
- Setting higher doesn't help - you still pay `dph_total`
- On-demand offers have fixed prices (not bids)

**Price Fluctuation Handling**:
- If price increases between search and provision, Vast.ai returns `{"success": False, "msg": "Offer no longer available"}`
- Broker should return `None` (operating error), caller can retry with different offer
- This matches RunPod/Lambda Labs pattern (see `broker/broker/api.py` lines 209+ for retry logic)

---

### Q4: GPU Filtering - Should we use Vast.ai's `contains` filter?

**Answer**: **Use hybrid approach**: Vast.ai filter for exact matches, Python filter for case-insensitive.

**Recommended Implementation** (balancing efficiency vs flexibility):

```python
def search_gpu_offers(..., gpu_type_hint: Optional[str] = None, ...):
    """
    Args:
        gpu_type_hint: Optional GPU type from ProvisionRequest.gpu_type
                      Used for API-level filtering if available
    """
    # Build base query
    query = {
        "verified": {"eq": True},
        "reliability2": {"gte": min_reliability},
        "num_gpus": {"eq": gpu_count},
        "order": [["dph_total", "asc"]],
    }

    # Attempt API-level filtering if gpu_type_hint provided
    # This reduces network transfer and response size
    if gpu_type_hint:
        # Normalize common variations for API query
        # Vast.ai uses exact model names: "RTX 4090", "H100 PCIe", "A100 SXM4"
        query["gpu_name"] = {"contains": gpu_type_hint}
        # NOTE: This is case-sensitive, so "h100" won't match "H100"

    response = _make_api_request("POST", "/bundles/", data=query, api_key=api_key)
    offers_raw = response.get("offers", [])

    # Post-filter in Python for flexibility (case-insensitive, partial match, etc.)
    offers = []
    for offer_data in offers_raw:
        gpu_name = offer_data.get("gpu_name", "").lower()

        # Apply manufacturer filter
        if manufacturer and manufacturer.lower() not in gpu_name:
            continue

        # Apply additional GPU type filters if needed
        # (This catches cases where API filter was too strict or case-mismatched)

        offers.append(GPUOffer(...))

    return offers
```

**Trade-off Analysis**:

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **API-only filter** | Fastest, minimal data transfer | Case-sensitive, inflexible | ❌ Too restrictive |
| **Python-only filter** | Flexible, case-insensitive | Large response size (10-100 MB) | ⚠️ Works but wasteful |
| **Hybrid (recommended)** | Balance of speed + flexibility | Slightly more complex code | ✅ Best of both |

**Why hybrid is better** (Casey Muratori - granularity):
- Provides both "high-level" (simple string) and "low-level" (custom filter) access
- Avoids integration discontinuity: user doesn't hit wall when needs change
- Degrades gracefully: if API filter too strict, Python filter catches missed results

---

### Q5: Error Recovery - Should integration test retry with different offers?

**Answer**: **Yes, retry with fallback** (matches existing pattern in `run_integration_test.py`).

**Current Pattern** (line 58-64):
```python
def provision_instance(gpu_client, offers, provider=None, ready_timeout=None, ssh_timeout=None):
    instance = gpu_client.create(
        offers,  # <-- Passes LIST of offers
        image="runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204",
        name="jax-integration-test",
        n_offers=len(offers)  # <-- Automatic fallback
    )
```

**How `GPUClient.create()` handles fallback**:
- Tries first offer → if fails, tries second → etc.
- Returns `None` only if ALL offers fail
- Each provider's `provision_instance()` returns `None` for operating errors

**Recommendation for Vast.ai**:
```python
# In broker/broker/api.py - create() function (line 209+)
def create(...):
    for offer in offers[:n_offers]:
        try:
            # Extract offer info
            offer_id = offer.id
            provider = offer.provider

            # Build provision request
            request = ProvisionRequest(...)

            # Try provisioning
            instance = provider_module.provision_instance(request, ...)

            if instance:
                logger.info(f"Successfully provisioned {instance.id}")
                return instance  # Success!
            else:
                logger.warning(f"Offer {offer_id} unavailable, trying next...")
                continue  # Try next offer

        except ValueError as e:
            logger.error(f"Programmer error: {e}")
            raise  # Don't retry programmer errors

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            continue  # Try next offer

    return None  # All offers failed
```

**Why this pattern?**
- Vast.ai offers go stale quickly (high churn)
- Integration test passes list of 5 cheapest offers
- Automatic fallback = robust test, matches RunPod behavior

---

### Q6: Manufacturer Filtering - Is Vast.ai really NVIDIA-only?

**Answer**: **No, Vast.ai supports AMD GPUs as of May 2024** (this assumption was outdated).

**Vast.ai Manufacturer Support** (as of 2024):
- **NVIDIA**: Full support (RTX, A100, H100, etc.)
- **AMD**: Radeon and Instinct product lines (added May 2024)
  - Renters should use ROCm template
  - Supported templates: Ubuntu 22.04, PyTorch

**Implementation Update**:

```python
def search_gpu_offers(..., manufacturer: Optional[str] = None, ...):
    # ... build query ...

    # Post-filter by manufacturer
    offers = []
    for offer_data in offers_raw:
        gpu_name = offer_data.get("gpu_name", "").lower()

        # Apply manufacturer filter (both NVIDIA and AMD supported)
        if manufacturer:
            mfr_lower = manufacturer.lower()

            # NVIDIA: Most GPUs (RTX, GTX, A100, H100, V100, etc.)
            is_nvidia = any(x in gpu_name for x in ["rtx", "gtx", "quadro", "tesla", "a100", "h100", "v100", "a6000", "a5000", "l40", "l4"])

            # AMD: Radeon and Instinct lines
            is_amd = any(x in gpu_name for x in ["radeon", "instinct", "rx ", "mi100", "mi200", "mi300"])

            # Filter based on manufacturer
            if mfr_lower == "nvidia" and not is_nvidia:
                continue
            elif mfr_lower == "amd" and not is_amd:
                continue
            elif mfr_lower not in ["nvidia", "amd"]:
                # Intel, other manufacturers not yet supported by Vast.ai
                continue

        offers.append(GPUOffer(...))

    return offers
```

**Documentation Update** (Section 5.5):
- ~~"Vast.ai is NVIDIA-only"~~ ← Remove this line
- Add: "Vast.ai supports NVIDIA (full lineup) and AMD (Radeon/Instinct) GPUs as of May 2024"
- Note: AMD GPUs require ROCm-compatible images (e.g., `rocm/pytorch`)

**Default Image Consideration**:
- Current default: `"runpod/pytorch:1.0.0-cu1281-torch280-ubuntu2204"` (CUDA/NVIDIA)
- For AMD GPUs, user must override with ROCm image
- Could add logic to auto-select image based on GPU manufacturer

---

**Document Version**: 1.1
**Last Updated**: 2025-01-29
**Author**: Claude Code (based on vast-python CLI analysis)
