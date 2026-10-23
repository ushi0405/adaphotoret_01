// Bundled inline by Streamlit v2: storage belongs to this app's stable origin,
// not a hashed component iframe, shared server directory, cookie or URL token.
const STORAGE_KEY = "adaphotoret.photo-library.v1";
const MAX_BYTES = 2 * 1024 * 1024;
const EMPTY = () => ({photo_tags: {}, libraries: {}});

async function revision(raw) {
  if (raw === null) return null;
  const hash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(raw));
  return [...new Uint8Array(hash)].map(value => value.toString(16).padStart(2, "0")).join("");
}

function decode(raw) {
  if (raw === null) return EMPTY();
  if (new TextEncoder().encode(raw).length > MAX_BYTES) throw new Error("invalid");
  const value = JSON.parse(raw);
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("invalid");
  // Raw metadata is the legacy format. Python performs strict field validation.
  if (!Object.hasOwn(value, "version")) return value;
  if (value.version !== 1 || Object.keys(value).sort().join(",") !== "data,version") throw new Error("invalid");
  return value.data;
}

export default function(component) {
  const {data, parentElement, setStateValue} = component;
  const request = data || {op: "watch"};
  let disposed = false;
  let readGeneration = 0;
  const emit = value => {
    // A request can outlive a rerender. Its write still completes safely, and
    // the next render retries the same idempotent request to recover its ack.
    if (!disposed) setStateValue("event", {event_id: crypto.randomUUID(), ...value});
  };
  const failure = error => {
    const code = error?.name === "QuotaExceededError" ? "quota" :
      error?.message === "invalid" || error instanceof SyntaxError ? "invalid" :
      error?.message === "unsupported" ? "unsupported" : "unavailable";
    emit({kind: "error", request_id: request.id, code});
  };
  const read = async kind => {
    const generation = ++readGeneration;
    const raw = localStorage.getItem(STORAGE_KEY);
    const value = decode(raw);
    const currentRevision = await revision(raw);
    if (generation === readGeneration) emit({kind, request_id: request.id, data: value, revision: currentRevision});
  };

  const run = async () => {
    if (request.op === "load") return read("loaded");
    // Reconcile any update between the previous acknowledgment and this watch.
    if (request.op === "watch") return read("external");
    if (request.op !== "save") return;
    // Require a real cross-tab transaction lock. A read-then-write alone can
    // race across renderer processes and resurrect a deleted album.
    if (!navigator.locks) throw new Error("unsupported");
    await navigator.locks.request(STORAGE_KEY, async () => {
      const next = JSON.stringify({version: 1, data: request.data});
      if (new TextEncoder().encode(next).length > MAX_BYTES) throw new Error("invalid");
      const raw = localStorage.getItem(STORAGE_KEY);
      // Replaying a write after an interrupted acknowledgment is harmless.
      if (raw === next) {
        emit({kind: "saved", request_id: request.id, revision: await revision(raw)});
        return;
      }
      if (await revision(raw) !== request.base_revision) {
        emit({kind: "conflict", request_id: request.id});
        return;
      }
      localStorage.setItem(STORAGE_KEY, next); // atomic; quota errors keep old data
      if (localStorage.getItem(STORAGE_KEY) !== next) throw new Error("unavailable");
      emit({kind: "saved", request_id: request.id, revision: await revision(next)});
    });
  };

  // The DOM marker prevents duplicate requests on ordinary widget reruns.
  if (request.id && parentElement.__adaCompletedRequest !== request.id) {
    run().then(() => {
      if (!disposed) parentElement.__adaCompletedRequest = request.id;
    }).catch(failure);
  }
  const onStorage = event => {
    if (event.key === STORAGE_KEY || event.key === null) read("external").catch(failure);
  };
  // Never overwrite a load/save acknowledgment with an external update before
  // Python consumes it; component state events can be coalesced by the client.
  if (request.op === "watch" && request.id) window.addEventListener("storage", onStorage);
  return () => { disposed = true; window.removeEventListener("storage", onStorage); };
}
