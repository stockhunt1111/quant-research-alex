// What the page reads, kept by TanStack Query until the server says it changed (live/useLive.ts): nothing polls.
import { keepPreviousData, queryOptions, useQuery } from "@tanstack/react-query";
import { get } from "./client";
import type { AssetRow, Curves, ListRow, Meta, ResultView, RunCurrent } from "./types";

export const useMeta = () => useQuery({ queryKey: ["meta"], queryFn: () => get<Meta>("/api/research/meta") });

export const useListRows = (enabled: boolean) =>
  useQuery({ queryKey: ["rows", "lists"], queryFn: () => get<{ rows: ListRow[] }>("/api/research/lists"), enabled });

export const useAssetRows = (enabled: boolean) =>
  useQuery({ queryKey: ["rows", "assets"], queryFn: () => get<{ rows: AssetRow[] }>("/api/research/assets"), enabled });

export const resultQuery = (key: string) =>
  queryOptions({ queryKey: ["result", key], queryFn: () => get<ResultView>("/api/research/result?key=" + encodeURIComponent(key)) });

export const useResult = (key: string) => useQuery(resultQuery(key));

export const useCurves = (keys: string[]) => {
  const sorted = keys.toSorted();
  return useQuery({
    queryKey: ["curves", ...sorted],
    queryFn: () => get<Curves>("/api/research/curves?" + sorted.map((k) => "keys=" + encodeURIComponent(k)).join("&")),
    enabled: sorted.length > 0,
    placeholderData: keepPreviousData,
  });
};

export const useRun = () => useQuery({ queryKey: ["run"], queryFn: () => get<RunCurrent>("/api/runs/current") });
