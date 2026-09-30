import React from "react";
import { createRoot } from "react-dom/client";
import { DesignPlatform } from "../../app/components/DesignPlatform";
import "../../app/globals.css";

const language = window.location.pathname === "/zh" || window.location.pathname.startsWith("/zh/") ? "zh" : "en";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode><DesignPlatform language={language} /></React.StrictMode>,
);
