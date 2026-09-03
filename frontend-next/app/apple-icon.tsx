import { ImageResponse } from "next/og";

export const size = { width: 180, height: 180 };
export const contentType = "image/png";

export default function AppleIcon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "linear-gradient(135deg, #EE0000 0%, #A30000 100%)",
          borderRadius: 36,
        }}
      >
        <span
          style={{
            fontSize: 72,
            fontWeight: 800,
            color: "#FFFFFF",
            letterSpacing: -2,
          }}
        >
          IDP
        </span>
      </div>
    ),
    { ...size },
  );
}
