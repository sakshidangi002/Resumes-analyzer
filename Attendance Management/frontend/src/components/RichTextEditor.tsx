import { useEffect, useRef } from "react";

const BTN: React.CSSProperties = {
  background: "rgba(255,255,255,0.08)",
  border: "1px solid rgba(255,255,255,0.15)",
  color: "#fff",
  borderRadius: 6,
  padding: "4px 10px",
  cursor: "pointer",
  fontSize: "0.85rem",
  lineHeight: 1,
};

/**
 * Lightweight rich-text editor (no external deps). Supports bold / italic /
 * underline / font size / bullet list via the browser's execCommand and emits
 * HTML. The initial value is applied once on mount to avoid caret jumps; give
 * the component a changing `key` to reset it (e.g. when opening a fresh form).
 */
export default function RichTextEditor({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (html: string) => void;
  placeholder?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current && ref.current.innerHTML !== (value || "")) {
      ref.current.innerHTML = value || "";
    }
    // Only on mount — controlled sync would fight the caret.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const exec = (cmd: string, arg?: string) => {
    ref.current?.focus();
    document.execCommand(cmd, false, arg);
    if (ref.current) onChange(ref.current.innerHTML);
  };

  return (
    <div style={{ border: "1px solid rgba(255,255,255,0.15)", borderRadius: 8, overflow: "hidden" }}>
      <div
        style={{
          display: "flex",
          gap: 6,
          flexWrap: "wrap",
          padding: 6,
          background: "rgba(255,255,255,0.04)",
          borderBottom: "1px solid rgba(255,255,255,0.1)",
        }}
      >
        {/* preventDefault on mousedown keeps the text selection while clicking */}
        <button type="button" style={{ ...BTN, fontWeight: 800 }} onMouseDown={(e) => e.preventDefault()} onClick={() => exec("bold")} title="Bold">B</button>
        <button type="button" style={{ ...BTN, fontStyle: "italic" }} onMouseDown={(e) => e.preventDefault()} onClick={() => exec("italic")} title="Italic">I</button>
        <button type="button" style={{ ...BTN, textDecoration: "underline" }} onMouseDown={(e) => e.preventDefault()} onClick={() => exec("underline")} title="Underline">U</button>
        <select
          defaultValue=""
          onMouseDown={(e) => e.stopPropagation()}
          onChange={(e) => {
            if (e.target.value) {
              exec("fontSize", e.target.value);
              e.target.value = "";
            }
          }}
          style={{ ...BTN }}
          title="Text size"
        >
          <option value="">Size ▾</option>
          <option value="2">Small</option>
          <option value="3">Normal</option>
          <option value="5">Large</option>
          <option value="6">Heading</option>
        </select>
        <button type="button" style={BTN} onMouseDown={(e) => e.preventDefault()} onClick={() => exec("insertUnorderedList")} title="Bullet list">• List</button>
      </div>
      <div
        ref={ref}
        contentEditable
        suppressContentEditableWarning
        onInput={() => ref.current && onChange(ref.current.innerHTML)}
        data-placeholder={placeholder || ""}
        style={{ minHeight: 140, padding: "0.6rem 0.8rem", outline: "none", color: "rgba(255,255,255,0.9)", lineHeight: 1.5 }}
      />
    </div>
  );
}
