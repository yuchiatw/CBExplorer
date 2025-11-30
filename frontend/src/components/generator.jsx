export default function Generator() {
    return (
        <div
            role="img"
            aria-label="Generator plot placeholder"
            style={{
                border: "2px dashed #d1d5db",
                borderRadius: 8,
                padding: 16,
                minHeight: 220,
                width: 500,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                background: "linear-gradient(90deg, #ffffff 0%, #f8fafc 100%)",
                color: "#6b7280",
                fontSize: 14,
            }}
        >
            <div style={{ textAlign: "center" }}>
                <strong>Plot Placeholder</strong>
                <div style={{ marginTop: 8 }}>Generator visualization will render here.</div>
            </div>
        </div>
    )
}