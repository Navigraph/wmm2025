/*--------------------------------------------------------------------------*/
/*
 * Fast WMM point / batch evaluation for the Python wrapper.
 *
 * Loads WMM.COF once, reuses Legendre / spherical-harmonic scratch buffers,
 * and skips secular-variation work that the Python API does not return.
 *
 * The Legendre functions and the (a/r)^(n+2) powers depend only on latitude
 * and altitude, so grid evaluation hoists them out of the longitude loop.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#include "GeomagnetismHeader.h"
#include "EGM9615.h"

/*
 * NOT thread-safe: the scratch buffers below are mutated by every evaluation,
 * so callers must serialize entry points (the Python wrapper holds a lock) or
 * use one process per thread of execution.
 */
typedef struct {
    int ready;
    int nMax;
    MAGtype_MagneticModel *MagneticModel;
    MAGtype_MagneticModel *TimedMagneticModel;
    MAGtype_Ellipsoid Ellip;
    MAGtype_Geoid Geoid;
    MAGtype_LegendreFunction *LegendreFunction;
    MAGtype_SphericalHarmonicVariables *SphVariables;
    double *schmidtQuasiNorm; /* Gauss -> Schmidt ratios, depend only on nMax */
    double cached_year;
    int year_valid;
} WMMState;

static WMMState g_wmm = {0};

static void wmm_clear(void)
{
    if(g_wmm.TimedMagneticModel) {
        MAG_FreeMagneticModelMemory(g_wmm.TimedMagneticModel);
        g_wmm.TimedMagneticModel = NULL;
    }
    if(g_wmm.MagneticModel) {
        MAG_FreeMagneticModelMemory(g_wmm.MagneticModel);
        g_wmm.MagneticModel = NULL;
    }
    if(g_wmm.LegendreFunction) {
        MAG_FreeLegendreMemory(g_wmm.LegendreFunction);
        g_wmm.LegendreFunction = NULL;
    }
    if(g_wmm.SphVariables) {
        MAG_FreeSphVarMemory(g_wmm.SphVariables);
        g_wmm.SphVariables = NULL;
    }
    free(g_wmm.schmidtQuasiNorm);
    g_wmm.schmidtQuasiNorm = NULL;
    g_wmm.ready = 0;
    g_wmm.year_valid = 0;
}

void wmm_free(void)
{
    wmm_clear();
}

/* Same ratios MAG_PcupLow derives internally; they only depend on n and m. */
static void wmm_build_schmidt_norm(double *snorm, int nMax)
{
    int n, m, index, index1;

    snorm[0] = 1.0;
    for(n = 1; n <= nMax; n++) {
        index = (n * (n + 1) / 2);
        index1 = (n - 1) * n / 2;
        snorm[index] = snorm[index1] * (double) (2 * n - 1) / (double) n;

        for(m = 1; m <= n; m++) {
            index = (n * (n + 1) / 2 + m);
            index1 = (n * (n + 1) / 2 + m - 1);
            snorm[index] = snorm[index1] *
                    sqrt((double) ((n - m + 1) * (m == 1 ? 2 : 1)) / (double) (n + m));
        }
    }
}

/*
 * MAG_PcupLow without the per-call malloc and without rebuilding the
 * normalization ratios. Valid for nMax <= 16, which covers WMM (nMax = 12).
 */
static void wmm_pcup_low(double *Pcup, double *dPcup, double x, int nMax,
                         const double *snorm)
{
    int n, m, index, index1, index2;
    double k, z;

    Pcup[0] = 1.0;
    dPcup[0] = 0.0;
    z = sqrt((1.0 - x) * (1.0 + x));

    for(n = 1; n <= nMax; n++) {
        for(m = 0; m <= n; m++) {
            index = (n * (n + 1) / 2 + m);
            if(n == m) {
                index1 = (n - 1) * n / 2 + m - 1;
                Pcup[index] = z * Pcup[index1];
                dPcup[index] = z * dPcup[index1] + x * Pcup[index1];
            } else if(n == 1 && m == 0) {
                index1 = (n - 1) * n / 2 + m;
                Pcup[index] = x * Pcup[index1];
                dPcup[index] = x * dPcup[index1] - z * Pcup[index1];
            } else if(n > 1 && n != m) {
                index1 = (n - 2) * (n - 1) / 2 + m;
                index2 = (n - 1) * n / 2 + m;
                if(m > n - 2) {
                    Pcup[index] = x * Pcup[index2];
                    dPcup[index] = x * dPcup[index2] - z * Pcup[index2];
                } else {
                    k = (double) (((n - 1) * (n - 1)) - (m * m)) /
                            (double) ((2 * n - 1) * (2 * n - 3));
                    Pcup[index] = x * Pcup[index2] - k * Pcup[index1];
                    dPcup[index] = x * dPcup[index2] - z * Pcup[index2] - k * dPcup[index1];
                }
            }
        }
    }

    for(n = 1; n <= nMax; n++) {
        for(m = 0; m <= n; m++) {
            index = (n * (n + 1) / 2 + m);
            Pcup[index] = Pcup[index] * snorm[index];
            /* Sign flip: derivative is with respect to latitude, not co-latitude. */
            dPcup[index] = -dPcup[index] * snorm[index];
        }
    }
}

int wmm_init(const char *cof_path)
{
    MAGtype_MagneticModel *models[1];
    char filename[4096];
    int nMax = 0;
    int NumTerms;

    if(!cof_path || !cof_path[0]) {
        fprintf(stderr, "wmm_init: empty coefficient path\n");
        return EXIT_FAILURE;
    }
    if(strlen(cof_path) >= sizeof(filename)) {
        fprintf(stderr, "wmm_init: coefficient path too long\n");
        return EXIT_FAILURE;
    }

    wmm_clear();

    strncpy(filename, cof_path, sizeof(filename) - 1);
    filename[sizeof(filename) - 1] = '\0';

    if(!MAG_robustReadMagModels(filename, &models, 1)) {
        fprintf(stderr, "wmm_init: WMM coefficients not found: %s\n", cof_path);
        return EXIT_FAILURE;
    }

    g_wmm.MagneticModel = models[0];
    if(nMax < g_wmm.MagneticModel->nMax)
        nMax = g_wmm.MagneticModel->nMax;
    NumTerms = ((nMax + 1) * (nMax + 2) / 2);

    g_wmm.nMax = nMax;
    g_wmm.TimedMagneticModel = MAG_AllocateModelMemory(NumTerms);
    g_wmm.LegendreFunction = MAG_AllocateLegendreFunctionMemory(NumTerms);
    g_wmm.SphVariables = MAG_AllocateSphVarMemory(nMax);
    g_wmm.schmidtQuasiNorm = (double *) malloc((NumTerms + 1) * sizeof(double));

    if(!g_wmm.MagneticModel || !g_wmm.TimedMagneticModel ||
       !g_wmm.LegendreFunction || !g_wmm.SphVariables || !g_wmm.schmidtQuasiNorm) {
        wmm_clear();
        MAG_Error(2);
        return EXIT_FAILURE;
    }

    wmm_build_schmidt_norm(g_wmm.schmidtQuasiNorm, nMax);

    MAG_SetDefaults(&g_wmm.Ellip, &g_wmm.Geoid);
    g_wmm.Geoid.GeoidHeightBuffer = GeoidHeightBuffer;
    g_wmm.Geoid.Geoid_Initialized = 1;
    g_wmm.Geoid.UseGeoid = 0;

    g_wmm.ready = 1;
    g_wmm.year_valid = 0;
    return EXIT_SUCCESS;
}

static void wmm_ensure_year(double yeardecimal)
{
    MAGtype_Date UserDate;

    if(g_wmm.year_valid && g_wmm.cached_year == yeardecimal)
        return;

    UserDate.DecimalYear = yeardecimal;
    UserDate.Year = 0;
    UserDate.Month = 0;
    UserDate.Day = 0;
    MAG_TimelyModifyMagneticModel(UserDate, g_wmm.MagneticModel, g_wmm.TimedMagneticModel);
    g_wmm.cached_year = yeardecimal;
    g_wmm.year_valid = 1;
}

/*
 * Everything that depends on latitude and altitude but not on longitude:
 * the spherical coordinates, the Legendre functions and (a/r)^(n+2).
 */
static void wmm_begin_row(double geolatitude, double alt_km,
                          MAGtype_CoordGeodetic *geo, MAGtype_CoordSpherical *sph)
{
    int nMax = g_wmm.nMax;
    double sin_phi, ratio;
    int n;

    geo->HeightAboveEllipsoid = alt_km;
    geo->HeightAboveGeoid = alt_km;
    geo->phi = geolatitude;
    geo->lambda = 0.0;
    geo->UseGeoid = 0;

    MAG_GeodeticToSpherical(g_wmm.Ellip, *geo, sph);

    sin_phi = sin(DEG2RAD(sph->phig));
    if(nMax <= 16 || (1 - fabs(sin_phi)) < 1.0e-10)
        wmm_pcup_low(g_wmm.LegendreFunction->Pcup, g_wmm.LegendreFunction->dPcup,
                     sin_phi, nMax, g_wmm.schmidtQuasiNorm);
    else
        MAG_AssociatedLegendreFunction(*sph, nMax, g_wmm.LegendreFunction);

    ratio = g_wmm.Ellip.re / sph->r;
    g_wmm.SphVariables->RelativeRadiusPower[0] = ratio * ratio;
    for(n = 1; n <= nMax; n++)
        g_wmm.SphVariables->RelativeRadiusPower[n] =
                g_wmm.SphVariables->RelativeRadiusPower[n - 1] * ratio;
}

/* Longitude-dependent part, given a row prepared by wmm_begin_row(). */
static void wmm_eval_in_row(double geolongitude,
                            MAGtype_CoordGeodetic *geo, MAGtype_CoordSpherical *sph,
                            double *X, double *Y, double *Z, double *F,
                            double *Decl, double *Incl)
{
    MAGtype_MagneticResults MagneticResultsSph, MagneticResultsGeo;
    MAGtype_GeoMagneticElements GeoMagneticElements;
    double cos_lambda, sin_lambda;
    int m, nMax = g_wmm.nMax;

    geo->lambda = geolongitude;
    sph->lambda = geolongitude;

    cos_lambda = cos(DEG2RAD(geolongitude));
    sin_lambda = sin(DEG2RAD(geolongitude));
    g_wmm.SphVariables->cos_mlambda[0] = 1.0;
    g_wmm.SphVariables->sin_mlambda[0] = 0.0;
    g_wmm.SphVariables->cos_mlambda[1] = cos_lambda;
    g_wmm.SphVariables->sin_mlambda[1] = sin_lambda;
    for(m = 2; m <= nMax; m++) {
        g_wmm.SphVariables->cos_mlambda[m] =
                g_wmm.SphVariables->cos_mlambda[m - 1] * cos_lambda -
                g_wmm.SphVariables->sin_mlambda[m - 1] * sin_lambda;
        g_wmm.SphVariables->sin_mlambda[m] =
                g_wmm.SphVariables->cos_mlambda[m - 1] * sin_lambda +
                g_wmm.SphVariables->sin_mlambda[m - 1] * cos_lambda;
    }

    MAG_Summation(g_wmm.LegendreFunction, g_wmm.TimedMagneticModel, *g_wmm.SphVariables,
                  *sph, &MagneticResultsSph);
    MAG_RotateMagneticVector(*sph, *geo, MagneticResultsSph, &MagneticResultsGeo);
    MAG_CalculateGeoMagneticElements(&MagneticResultsGeo, &GeoMagneticElements);

    *X = GeoMagneticElements.X;
    *Y = GeoMagneticElements.Y;
    *Z = GeoMagneticElements.Z;
    *F = GeoMagneticElements.F;
    *Decl = GeoMagneticElements.Decl;
    *Incl = GeoMagneticElements.Incl;
}

static void wmm_eval_into(double geolatitude, double geolongitude,
                          double alt_km, double yeardecimal,
                          double *X, double *Y, double *Z, double *F,
                          double *Decl, double *Incl)
{
    MAGtype_CoordGeodetic geo;
    MAGtype_CoordSpherical sph;

    wmm_ensure_year(yeardecimal);
    wmm_begin_row(geolatitude, alt_km, &geo, &sph);
    wmm_eval_in_row(geolongitude, &geo, &sph, X, Y, Z, F, Decl, Incl);
}

int wmm_eval(double geolatitude, double geolongitude,
             double HeightAboveEllipsoid, double yeardecimal,
             double *X, double *Y, double *Z, double *F,
             double *Decl, double *Incl)
{
    if(!g_wmm.ready)
        return EXIT_FAILURE;

    wmm_eval_into(geolatitude, geolongitude, HeightAboveEllipsoid, yeardecimal,
                  X, Y, Z, F, Decl, Incl);
    return EXIT_SUCCESS;
}

/*
 * Outer-product grid: nlat x nlon points in row-major order, one altitude and
 * one epoch. The Legendre functions are computed once per latitude row.
 */
int wmm_eval_latlon_grid(const double *lats, int nlat, const double *lons, int nlon,
                         double alt_km, double yeardecimal,
                         double *X, double *Y, double *Z, double *F,
                         double *Decl, double *Incl)
{
    MAGtype_CoordGeodetic geo;
    MAGtype_CoordSpherical sph;
    int i, j;

    if(!g_wmm.ready || nlat < 0 || nlon < 0 || !lats || !lons ||
       !X || !Y || !Z || !F || !Decl || !Incl)
        return EXIT_FAILURE;

    wmm_ensure_year(yeardecimal);

    for(i = 0; i < nlat; i++) {
        size_t base = (size_t) i * (size_t) nlon;

        wmm_begin_row(lats[i], alt_km, &geo, &sph);
        for(j = 0; j < nlon; j++) {
            size_t k = base + (size_t) j;
            wmm_eval_in_row(lons[j], &geo, &sph,
                            &X[k], &Y[k], &Z[k], &F[k], &Decl[k], &Incl[k]);
        }
    }
    return EXIT_SUCCESS;
}

/* Arbitrary point list at constant altitude and epoch. */
int wmm_eval_grid(const double *glats, const double *glons, int n,
                  double alt_km, double yeardecimal,
                  double *X, double *Y, double *Z, double *F,
                  double *Decl, double *Incl)
{
    MAGtype_CoordGeodetic geo;
    MAGtype_CoordSpherical sph;
    int i;

    if(!g_wmm.ready || n < 0 || !glats || !glons ||
       !X || !Y || !Z || !F || !Decl || !Incl)
        return EXIT_FAILURE;

    wmm_ensure_year(yeardecimal);

    for(i = 0; i < n; i++) {
        /* Consecutive points often share a latitude; only redo the row when it changes. */
        if(i == 0 || glats[i] != glats[i - 1])
            wmm_begin_row(glats[i], alt_km, &geo, &sph);
        wmm_eval_in_row(glons[i], &geo, &sph,
                        &X[i], &Y[i], &Z[i], &F[i], &Decl[i], &Incl[i]);
    }
    return EXIT_SUCCESS;
}

/* Per-point altitude and year (transects / profiles). */
int wmm_eval_many(const double *glats, const double *glons,
                  const double *alt_km, const double *yeardecimal, int n,
                  double *X, double *Y, double *Z, double *F,
                  double *Decl, double *Incl)
{
    int i;

    if(!g_wmm.ready || n < 0 || !glats || !glons || !alt_km || !yeardecimal ||
       !X || !Y || !Z || !F || !Decl || !Incl)
        return EXIT_FAILURE;

    for(i = 0; i < n; i++) {
        wmm_eval_into(glats[i], glons[i], alt_km[i], yeardecimal[i],
                      &X[i], &Y[i], &Z[i], &F[i], &Decl[i], &Incl[i]);
    }
    return EXIT_SUCCESS;
}

/* Backward-compatible single-shot API (still avoids chdir if path is absolute). */
int wmmsub(double geolatitude, double geolongitude, double HeightAboveEllipsoid, double yeardecimal,
           double *X, double *Y, double *Z, double *F, double *Decl, double *Incl)
{
    if(!g_wmm.ready) {
        /* Fall back to relative WMM.COF for legacy callers that chdir first. */
        if(wmm_init("WMM.COF") != EXIT_SUCCESS)
            return EXIT_FAILURE;
    }
    return wmm_eval(geolatitude, geolongitude, HeightAboveEllipsoid, yeardecimal,
                    X, Y, Z, F, Decl, Incl);
}
