const MAP_RADIUS_METRES = 90;
const OSM_ENDPOINT = "https://api.openstreetmap.org/api/0.6/map";

export default async function handler(request, response) {
  const lat = Number(request.query?.lat);
  const lon = Number(request.query?.lon);

  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    return response.status(400).json({ error: "Valid lat and lon are required" });
  }

  const latitudeRadius = MAP_RADIUS_METRES / 110540;
  const longitudeRadius = MAP_RADIUS_METRES /
    (111320 * Math.max(Math.cos((lat * Math.PI) / 180), 0.01));
  const bbox = [
    lon - longitudeRadius,
    lat - latitudeRadius,
    lon + longitudeRadius,
    lat + latitudeRadius,
  ].join(",");

  try {
    const upstream = await fetch(`${OSM_ENDPOINT}?bbox=${bbox}`, {
      headers: {
        Accept: "application/xml",
        "User-Agent": "Arch-EPVC/1.0 (school-building research platform)",
      },
      signal: AbortSignal.timeout(15000),
    });

    if (!upstream.ok) {
      throw new Error(`OpenStreetMap returned HTTP ${upstream.status}`);
    }

    const xml = await upstream.text();
    if (!xml.includes("<osm")) {
      throw new Error("OpenStreetMap returned no XML map data");
    }

    response.setHeader("Content-Type", "application/xml; charset=utf-8");
    response.setHeader("Cache-Control", "public, s-maxage=300, stale-while-revalidate=3600");
    return response.status(200).send(xml);
  } catch (error) {
    return response.status(502).json({
      error: error instanceof Error ? error.message : "OpenStreetMap request failed",
    });
  }
}
