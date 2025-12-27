# Image Viewing Support for Read Tool

**DRI:**
**Claude:** [this conversation]

## Context

Add image viewing capability to the `read` tool so the agent can view screenshots, diagrams, and other images inline. Currently, attempting to read an image file fails with "Cannot read binary file".

## Out of Scope

- Image generation/editing
- Video support
- PDF rendering
- Remote image URLs (already supported via ImageContent)

## Current State

### ImageContent has wrong fields

```python
# rollouts/dtypes.py - CURRENT (broken)
@dataclass(frozen=True)
class ImageContent(JsonSerializable):
    """Image content block in a message (for vision models)."""
    type: Literal["image"] = "image"
    image_url: str = ""  # ❌ Wrong: providers expect `data` and `mime_type`
    detail: str | None = None
```

### Providers already expect correct fields

```python
# rollouts/providers/anthropic.py - lines 132-150
elif isinstance(block, ImageContent):
    if block.data.startswith("http"):  # ❌ AttributeError: no `data` field
        content_parts.append({
            "type": "image",
            "source": {"type": "url", "url": block.data},
        })
    else:
        content_parts.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": block.mime_type,  # ❌ AttributeError: no `mime_type` field
                "data": block.data,
            },
        })

# rollouts/providers/openai_completions.py - lines 95-101
elif isinstance(block, ImageContent):
    if block.data.startswith("http"):  # ❌ AttributeError
        url = block.data
    else:
        url = f"data:{block.mime_type};base64,{block.data}"  # ❌ AttributeError
    content_parts.append({"type": "image_url", "image_url": {"url": url}})
```

### Read tool rejects binary files

```python
# rollouts/environments/coding.py - lines 685-691
try:
    content = abs_path.read_text(encoding="utf-8")
except UnicodeDecodeError:
    return ToolResult(
        tool_call_id=tool_call.id,
        is_error=True,
        content="",
        error=f"Cannot read binary file: {path_str}",  # ❌ Images fail here
    )
```

## Solution

### 1. Fix ImageContent in dtypes.py

Rename `image_url` to `data`, add `mime_type` field to match what providers expect:

```python
# rollouts/dtypes.py - FIXED
@dataclass(frozen=True)
class ImageContent(JsonSerializable):
    """Image content block in a message (for vision models).
    
    For base64 images: data contains base64 string, mime_type is set
    For URL images: data contains HTTP URL, mime_type can be None
    """
    type: Literal["image"] = "image"
    data: str = ""  # Base64 encoded image data OR HTTP URL
    mime_type: str | None = None  # e.g., "image/png", "image/jpeg"
    detail: str | None = None  # OpenAI detail parameter: "low", "high", "auto"
```

### 2. Add image detection using magic bytes

Reference implementation from pi-mono (`/tmp/pi-mono/packages/coding-agent/src/utils/mime.ts`):

```typescript
// pi-mono detects images by reading first 4100 bytes and checking magic bytes
const FILE_TYPE_SNIFF_BYTES = 4100;
const IMAGE_MIME_TYPES = new Set(["image/jpeg", "image/png", "image/gif", "image/webp"]);

const fileType = await fileTypeFromBuffer(buffer.subarray(0, bytesRead));
if (fileType && IMAGE_MIME_TYPES.has(fileType.mime)) {
    return fileType.mime;
}
```

Python implementation for rollouts:

```python
# rollouts/environments/coding.py - NEW

# Supported image MIME types for vision models
SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png", 
    "image/gif",
    "image/webp",
}

# Magic byte signatures for image detection
# Format: (offset, magic_bytes, mime_type)
IMAGE_SIGNATURES = [
    (0, b'\x89PNG\r\n\x1a\n', 'image/png'),
    (0, b'\xff\xd8\xff', 'image/jpeg'),
    (0, b'GIF87a', 'image/gif'),
    (0, b'GIF89a', 'image/gif'),
    (0, b'RIFF', 'image/webp'),  # RIFF....WEBP (check WEBP at offset 8)
]

def detect_image_mime_type(file_path: Path) -> str | None:
    """Detect if file is a supported image by checking magic bytes.
    
    Returns mime type string if supported image, None otherwise.
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)  # Read enough for all signatures
            
        if len(header) < 4:
            return None
            
        # Check each signature
        for offset, magic, mime_type in IMAGE_SIGNATURES:
            if header[offset:offset + len(magic)] == magic:
                # Special case: WEBP needs additional check
                if mime_type == 'image/webp':
                    if len(header) >= 12 and header[8:12] == b'WEBP':
                        return mime_type
                else:
                    return mime_type
                    
        return None
    except (OSError, IOError):
        return None
```

### 3. Update read tool to return ImageContent for images

```python
# rollouts/environments/coding.py - MODIFIED _exec_read method

async def _exec_read(self, tool_call: ToolCall) -> ToolResult:
    """Read file contents. Returns ImageContent for supported image files."""
    path_str = tool_call.args["path"]
    offset = tool_call.args.get("offset")
    limit = tool_call.args.get("limit")

    abs_path = expand_path(path_str)

    if not abs_path.exists():
        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=True,
            content="",
            error=f"File not found: {path_str}",
        )

    if not abs_path.is_file():
        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=True,
            content="",
            error=f"Not a file: {path_str}",
        )

    # Check if file is a supported image
    mime_type = detect_image_mime_type(abs_path)
    if mime_type:
        # Read and return as ImageContent
        import base64
        image_data = abs_path.read_bytes()
        base64_data = base64.b64encode(image_data).decode('ascii')
        
        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=[
                TextContent(text=f"Read image file [{mime_type}]: {path_str}"),
                ImageContent(data=base64_data, mime_type=mime_type),
            ],
        )

    # Existing text file handling...
    try:
        content = abs_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=True,
            content="",
            error=f"Cannot read binary file: {path_str}",
        )

    # ... rest of text handling unchanged
```

### 4. Update read tool description

```python
# rollouts/environments/coding.py - TOOLS list

Tool(
    function=ToolFunction(
        name="read",
        description="Read the contents of a file. Supports text files and images (jpg, png, gif, webp). Images are returned as viewable attachments. For text files, output is truncated to 2000 lines. Use offset/limit for large files.",
        parameters=ToolFunctionParameter(
            properties={
                "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
                "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)"},
                "limit": {"type": "integer", "description": "Maximum number of lines to read"},
            }
        ),
        required=["path"],
    )
),
```

## Files Changed

| File | Change |
|------|--------|
| `rollouts/dtypes.py` | Fix `ImageContent`: rename `image_url` → `data`, add `mime_type` field |
| `rollouts/environments/coding.py` | Add `detect_image_mime_type()`, update `_exec_read()` to return `ImageContent` for images |

## Testing

### Unit tests

```python
# tests/test_image_support.py

import base64
import tempfile
from pathlib import Path

import pytest

from rollouts.environments.coding import detect_image_mime_type, CodingEnvironment
from rollouts.dtypes import ImageContent, TextContent, ToolCall


class TestImageDetection:
    """Test magic byte detection for image files."""

    def test_detect_png(self, tmp_path):
        # PNG magic bytes: 89 50 4E 47 0D 0A 1A 0A
        png_file = tmp_path / "test.png"
        png_file.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
        assert detect_image_mime_type(png_file) == "image/png"

    def test_detect_jpeg(self, tmp_path):
        # JPEG magic bytes: FF D8 FF
        jpeg_file = tmp_path / "test.jpg"
        jpeg_file.write_bytes(b'\xff\xd8\xff\xe0' + b'\x00' * 100)
        assert detect_image_mime_type(jpeg_file) == "image/jpeg"

    def test_detect_gif87a(self, tmp_path):
        gif_file = tmp_path / "test.gif"
        gif_file.write_bytes(b'GIF87a' + b'\x00' * 100)
        assert detect_image_mime_type(gif_file) == "image/gif"

    def test_detect_gif89a(self, tmp_path):
        gif_file = tmp_path / "test.gif"
        gif_file.write_bytes(b'GIF89a' + b'\x00' * 100)
        assert detect_image_mime_type(gif_file) == "image/gif"

    def test_detect_webp(self, tmp_path):
        # WEBP: RIFF....WEBP
        webp_file = tmp_path / "test.webp"
        webp_file.write_bytes(b'RIFF\x00\x00\x00\x00WEBP' + b'\x00' * 100)
        assert detect_image_mime_type(webp_file) == "image/webp"

    def test_detect_text_file(self, tmp_path):
        text_file = tmp_path / "test.txt"
        text_file.write_text("Hello, world!")
        assert detect_image_mime_type(text_file) is None

    def test_detect_python_file(self, tmp_path):
        py_file = tmp_path / "test.py"
        py_file.write_text("print('hello')")
        assert detect_image_mime_type(py_file) is None

    def test_detect_empty_file(self, tmp_path):
        empty_file = tmp_path / "empty"
        empty_file.write_bytes(b'')
        assert detect_image_mime_type(empty_file) is None

    def test_detect_nonexistent_file(self, tmp_path):
        assert detect_image_mime_type(tmp_path / "nonexistent") is None


class TestReadToolImages:
    """Test read tool returns ImageContent for images."""

    @pytest.fixture
    def env(self, tmp_path):
        return CodingEnvironment(cwd=str(tmp_path))

    @pytest.mark.trio
    async def test_read_png_returns_image_content(self, env, tmp_path):
        # Create a minimal valid PNG
        png_file = tmp_path / "test.png"
        png_file.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)

        tool_call = ToolCall(id="test-1", name="read", args={"path": str(png_file)})
        result = await env._exec_read(tool_call)

        assert not result.is_error
        assert isinstance(result.content, list)
        assert len(result.content) == 2
        assert isinstance(result.content[0], TextContent)
        assert "image/png" in result.content[0].text
        assert isinstance(result.content[1], ImageContent)
        assert result.content[1].mime_type == "image/png"
        # Verify base64 encoding
        decoded = base64.b64decode(result.content[1].data)
        assert decoded.startswith(b'\x89PNG')

    @pytest.mark.trio
    async def test_read_text_file_unchanged(self, env, tmp_path):
        text_file = tmp_path / "test.txt"
        text_file.write_text("Hello, world!")

        tool_call = ToolCall(id="test-2", name="read", args={"path": str(text_file)})
        result = await env._exec_read(tool_call)

        assert not result.is_error
        assert isinstance(result.content, str)
        assert "Hello, world!" in result.content

    @pytest.mark.trio
    async def test_read_unknown_binary_still_fails(self, env, tmp_path):
        # Binary file that's not a recognized image
        bin_file = tmp_path / "test.bin"
        bin_file.write_bytes(b'\x00\x01\x02\x03' * 100)

        tool_call = ToolCall(id="test-3", name="read", args={"path": str(bin_file)})
        result = await env._exec_read(tool_call)

        assert result.is_error
        assert "Cannot read binary file" in result.error
```

### Integration test

```python
# Test with real image and vision model
@pytest.mark.trio
async def test_agent_can_describe_image(tmp_path):
    """End-to-end: agent reads image and describes it."""
    # Create a simple test image (1x1 red pixel PNG)
    import base64
    red_pixel_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )
    (tmp_path / "red.png").write_bytes(red_pixel_png)

    env = CodingEnvironment(cwd=str(tmp_path))
    endpoint = Endpoint(provider="anthropic", model="claude-sonnet-4-20250514")
    
    # ... run agent with "describe the image at red.png" task
    # Verify assistant response mentions red/color
```

### Manual testing

```bash
# 1. Read a PNG file
$ python -c "
from rollouts.environments.coding import detect_image_mime_type
from pathlib import Path
print(detect_image_mime_type(Path('screenshot.png')))
"
# Expected: image/png

# 2. Test in TUI
$ python -m rollouts.frontends.tui
> read screenshot.png
# Expected: Shows "Read image file [image/png]" and image is sent to model
```

## Migration Notes

### Breaking change

`ImageContent.image_url` renamed to `ImageContent.data`. Any code creating `ImageContent` directly needs updating:

```python
# Before
ImageContent(image_url="https://example.com/image.png")
ImageContent(image_url=base64_data)

# After  
ImageContent(data="https://example.com/image.png")
ImageContent(data=base64_data, mime_type="image/png")
```

### Backwards compatibility

The providers already expect `data` and `mime_type`, so fixing `ImageContent` makes them work correctly. No provider changes needed.
