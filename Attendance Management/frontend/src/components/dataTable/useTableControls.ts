import { useMemo, useState, useCallback } from "react";

export type SortDirection = "asc" | "desc";

export interface SortState<K extends string = string> {
  key: K | null;
  direction: SortDirection;
}

export interface ColumnFilterState {
  [columnKey: string]: string;
}

export interface UseTableControlsOptions<T> {
  /** All rows from the backend (unfiltered). */
  rows: T[];
  /**
   * Function that returns the searchable text for a row.
   * Used by the global search box. Default: JSON.stringify(row).
   */
  searchableText?: (row: T) => string;
  /**
   * Per-column accessor map. Used for sorting and per-column filtering.
   * Key = column key, value = function that returns the raw cell value.
   */
  columns: Record<string, (row: T) => unknown>;
  /** Initial sort state. */
  initialSort?: SortState;
  /** Initial column filters (e.g. { status: "Active" }). */
  initialColumnFilters?: ColumnFilterState;
}

export interface UseTableControlsResult<T> {
  /** Filtered + sorted rows ready to render. */
  displayed: T[];
  /** Original (unfiltered) rows. */
  rows: T[];
  /** Global search text. */
  search: string;
  setSearch: (v: string) => void;
  /** Sort state + setter. Click headers via `toggleSort(key)`. */
  sort: SortState;
  setSort: (s: SortState) => void;
  toggleSort: (key: string) => void;
  /** Per-column filters (e.g. status dropdown). */
  columnFilters: ColumnFilterState;
  setColumnFilter: (key: string, value: string) => void;
  clearAll: () => void;
  /** True if any filter / sort / search is active. */
  hasActiveControls: boolean;
}

function normalize(value: unknown): string {
  if (value == null) return "";
  if (value instanceof Date) return value.toISOString();
  return String(value);
}

function compareValues(a: unknown, b: unknown): number {
  const aIsNum = typeof a === "number" && Number.isFinite(a);
  const bIsNum = typeof b === "number" && Number.isFinite(b);
  if (aIsNum && bIsNum) return (a as number) - (b as number);

  const aStr = normalize(a);
  const bStr = normalize(b);

  const aNum = Number(aStr);
  const bNum = Number(bStr);
  if (!Number.isNaN(aNum) && !Number.isNaN(bNum) && aStr !== "" && bStr !== "") {
    return aNum - bNum;
  }

  // Date strings (YYYY-MM-DD etc.)
  const aDate = Date.parse(aStr);
  const bDate = Date.parse(bStr);
  if (!Number.isNaN(aDate) && !Number.isNaN(bDate) && /\d{4}/.test(aStr) && /\d{4}/.test(bStr)) {
    return aDate - bDate;
  }

  return aStr.localeCompare(bStr, undefined, { sensitivity: "base", numeric: true });
}

export function useTableControls<T>(opts: UseTableControlsOptions<T>): UseTableControlsResult<T> {
  const { rows, searchableText, columns, initialSort, initialColumnFilters } = opts;
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<SortState>(initialSort ?? { key: null, direction: "asc" });
  const [columnFilters, setColumnFiltersState] = useState<ColumnFilterState>(initialColumnFilters ?? {});

  const setColumnFilter = useCallback((key: string, value: string) => {
    setColumnFiltersState((prev) => {
      const next = { ...prev };
      if (!value) delete next[key];
      else next[key] = value;
      return next;
    });
  }, []);

  const toggleSort = useCallback((key: string) => {
    setSort((prev) => {
      if (prev.key !== key) return { key, direction: "asc" };
      if (prev.direction === "asc") return { key, direction: "desc" };
      return { key: null, direction: "asc" };
    });
  }, []);

  const clearAll = useCallback(() => {
    setSearch("");
    setSort({ key: null, direction: "asc" });
    setColumnFiltersState({});
  }, []);

  const displayed = useMemo(() => {
    const lowerSearch = search.trim().toLowerCase();

    let result = rows;

    if (lowerSearch) {
      result = result.filter((row) => {
        const text = searchableText
          ? searchableText(row)
          : Object.values(columns)
              .map((accessor) => normalize(accessor(row)))
              .join(" ");
        return text.toLowerCase().includes(lowerSearch);
      });
    }

    const activeFilters = Object.entries(columnFilters).filter(([, v]) => v !== "" && v != null);
    if (activeFilters.length) {
      result = result.filter((row) =>
        activeFilters.every(([key, needle]) => {
          const accessor = columns[key];
          if (!accessor) return true;
          const cell = normalize(accessor(row)).toLowerCase();
          return cell.includes(needle.toLowerCase());
        })
      );
    }

    if (sort.key && columns[sort.key]) {
      const accessor = columns[sort.key];
      const dir = sort.direction === "asc" ? 1 : -1;
      result = [...result].sort((a, b) => compareValues(accessor(a), accessor(b)) * dir);
    }

    return result;
  }, [rows, search, columnFilters, sort, columns, searchableText]);

  const hasActiveControls =
    search !== "" || sort.key !== null || Object.values(columnFilters).some((v) => v !== "");

  return {
    displayed,
    rows,
    search,
    setSearch,
    sort,
    setSort,
    toggleSort,
    columnFilters,
    setColumnFilter,
    clearAll,
    hasActiveControls,
  };
}
