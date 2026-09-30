import type { Metadata } from "next";
import { DesignPlatform } from "../components/DesignPlatform";

export const metadata: Metadata = {
  title: "Arch-EPVC · 学校建筑性能智能评价",
  description: "面向学校建筑能耗、光伏发电与碳排放的多模态年度及逐时预测平台。",
  alternates: { canonical: "/zh", languages: { en: "/", "zh-CN": "/zh" } },
};

export default function ChineseMirrorPage(){
  return <DesignPlatform language="zh"/>;
}
