export type Thermal = { roofU: number | null; wallU: number | null; groundU: number | null; windowU: number | null; shgc: number | null; status: string };
export type Prediction = { trueEui?: number; annualEui: number; trueEpv?: number; annualEpv: number; trueCei?: number; annualCei: number; firstEui: number; firstEpv: number; firstCei: number; source?: string };
export type Building = {
  id: string; teachingId: string; campusId: string; city: string; school: string;
  campusBuildingNo: number; sourceBuildingIds: string | null; sourceBuildingCount: number;
  lat: number; lon: number; obj: string; length: number; width: number; height: number; floors: number;
  year: number | null; constructionYearRaw: string | null; shapeFactor: number; orientation: number; footprintArea: number; buildingArea: number; roofArea: number;
  southFacadeArea: number; wwr: number; roofPvArea: number; facadePvArea: number; enclosureType: string | null;
  confidence: string | null; filterReason: string | null; schoolType: string | null; averageFloorHeight: number | null;
  buildingSpacing: number | null; campusArea: number | null; campusDensity: number | null; campusFar: number | null;
  envelopeBucket: string | null; shapeBin: string | null; thermal: Thermal; prediction: Prediction | null; staticFeatureIndex?: number;
};
export type HourlyPoint = { hour: number; energy: number; pv: number; carbon: number };
export type HourlyPerformancePoint = HourlyPoint & {
  netCost: number; exportRevenue: number;
  pvWithout: number; pvFull: number;
  carbonWithoutPv: number; carbonWithPv: number;
};
export type EpwData = { name: string; location: string; rows: number; points: Array<{day:number;hour:number;temperature:number;humidity:number;wind:number;dni:number;dhi:number;ghi:number}> };
