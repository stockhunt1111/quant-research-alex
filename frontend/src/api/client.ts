// Requests to the server. A refusal (409) carries the server's reason and whether asking again with confirm goes ahead.
import type { Refusal } from "./types";

export class Refused extends Error {
  confirm: boolean;
  constructor(r: Refusal) {
    super(r.message);
    this.confirm = r.confirm;
  }
}

async function failure(res: Response): Promise<Error> {
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    /* not JSON: the status says it */
  }
  if (res.status === 409 && body && typeof body === "object" && "message" in body) return new Refused(body as Refusal);
  const detail = body && typeof body === "object" && "detail" in body ? String((body as { detail: unknown }).detail) : "";
  return new Error(`${res.status} ${res.statusText}${detail ? ": " + detail : ""}`);
}

export async function get<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw await failure(res);
  return (await res.json()) as T;
}

export async function post<T>(url: string, body: unknown = {}): Promise<T> {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) throw await failure(res);
  return (await res.json()) as T;
}
