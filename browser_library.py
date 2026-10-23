"""Dependency-free browser storage bridge, registered once per Python process."""
from pathlib import Path
from streamlit.components.v2 import component

_bridge = component(
    "adaphotoret_private_library",
    js=Path(__file__).with_suffix(".js").read_text(encoding="utf-8"),
)

# A native Streamlit download_button uses a server media URL. Keep this backup
# entirely in a browser-created Blob instead, with no public download endpoint.
_download = component(
    "adaphotoret_private_library_backup",
    html='<button type="button">导出照片库备份</button>',
    css="""button {padding: .5rem .8rem; border: 1px solid #7778; border-radius: .5rem;
        color: var(--st-text-color); background: var(--st-secondary-background-color);
        font: inherit; cursor: pointer;} button:hover {border-color: var(--st-primary-color);}""",
    js="""export default function({data, parentElement}) {
        const button = parentElement.querySelector('button');
        const download = () => {
            const url = URL.createObjectURL(new Blob([data], {type: 'application/json;charset=utf-8'}));
            const anchor = document.createElement('a');
            anchor.href = url; anchor.download = 'adaphotoret-photo-library.json';
            anchor.click();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        };
        button.addEventListener('click', download);
        return () => button.removeEventListener('click', download);
    }""",
)


def browser_library_bridge(request):
    result = _bridge(data=request, key="private_photo_library_bridge", height=0,
                     default={"event": None}, on_event_change=lambda: None)
    return result.event


def browser_library_download(backup, key="private_photo_library_backup"):
    _download(data=backup.decode("utf-8"), key=key)
