import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { SseProvider } from "./live/sse";
import { TipProvider } from "./ui/Tip";
import { ToastProvider } from "./ui/Toast";
import "./styles.css";

// what the page read stays until the server says it changed (live/useLive.ts): nothing refetches on a timer or on focus
const client = new QueryClient({
  defaultOptions: { queries: { staleTime: Infinity, gcTime: 10 * 60_000, retry: 1, refetchOnWindowFocus: false } },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <SseProvider url="/api/events">
        <TipProvider>
          <ToastProvider>
            <App />
          </ToastProvider>
        </TipProvider>
      </SseProvider>
    </QueryClientProvider>
  </StrictMode>,
);
