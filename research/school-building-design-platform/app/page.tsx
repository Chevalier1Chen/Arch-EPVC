import type { Metadata } from "next";
import { DesignPlatform } from "./components/DesignPlatform";

export const metadata: Metadata = {
  title: "Arch-EPVC · Building Performance Intelligence",
  description: "Multimodal annual and hourly prediction of school-building energy, photovoltaics and carbon emissions.",
};

export default function Home() {
  return <DesignPlatform language="en" />;
}
