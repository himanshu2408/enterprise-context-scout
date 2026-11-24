export interface SchemaIssue {
  path: string[];
  message: string;
}

export interface ParseResult<T> {
  success: boolean;
  data?: T;
  issues?: SchemaIssue[];
}

export function safeParseNonEmptyString(value: unknown): ParseResult<string> {
  if (typeof value !== "string" || value.trim().length === 0) {
    return {
      success: false,
      issues: [{ path: [], message: "Expected non-empty string" }],
    };
  }
  return { success: true, data: value.trim() };
}
