import { ImageResponse } from "next/og";

export const size = { width: 32, height: 32 };
export const contentType = "image/png";

export default function Icon() {
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
          borderRadius: 6,
        }}
      >
        <span
          style={{
            fontSize: 14,
            fontWeight: 800,
            color: "#FFFFFF",
            letterSpacing: -0.5,
          }}
        >
          IDP
        </span>
      </div>
    ),
    { ...size },
  );
}
