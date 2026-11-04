/**
 * Approximate solar position (good to a fraction of a degree, plenty for
 * lighting). Pure functions, no three.js: unit-testable under node.
 */

export interface SolarPosition {
  /** Radians clockwise from north. */
  azimuth: number;
  /** Radians above the horizon (negative = below). */
  elevation: number;
}

const DEG = Math.PI / 180;

/** Day of year (1..366) of a date, in UTC. */
export function dayOfYear(date: Date): number {
  const start = Date.UTC(date.getUTCFullYear(), 0, 1);
  return Math.floor((date.getTime() - start) / 86400000) + 1;
}

/**
 * Sun position for a latitude, day of year and *local solar* hour (12 = solar
 * noon). Longitude is not needed for solar time.
 */
export function solarPosition(latDeg: number, doy: number, solarHour: number): SolarPosition {
  const lat = latDeg * DEG;
  const decl = 23.44 * DEG * Math.sin((2 * Math.PI * (284 + doy)) / 365);
  const hourAngle = 15 * DEG * (solarHour - 12);
  const sinEl = Math.sin(lat) * Math.sin(decl) + Math.cos(lat) * Math.cos(decl) * Math.cos(hourAngle);
  const elevation = Math.asin(Math.max(-1, Math.min(1, sinEl)));
  const denom = Math.cos(elevation) * Math.cos(lat);
  let azimuth: number;
  if (Math.abs(denom) < 1e-9) {
    azimuth = Math.PI; // sun at zenith or observer at a pole
  } else {
    const cosAz = (Math.sin(decl) - Math.sin(elevation) * Math.sin(lat)) / denom;
    azimuth = Math.acos(Math.max(-1, Math.min(1, cosAz)));
    if (hourAngle > 0) azimuth = 2 * Math.PI - azimuth;
  }
  return { azimuth, elevation };
}

/** Unit vector towards the sun in local ENU (east, north, up). */
export function sunDirectionEnu(p: SolarPosition): [number, number, number] {
  const c = Math.cos(p.elevation);
  return [Math.sin(p.azimuth) * c, Math.cos(p.azimuth) * c, Math.sin(p.elevation)];
}
