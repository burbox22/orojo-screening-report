"""
generate_report.py
------------------
Runs T1-T5 screening pipeline for a given bbox and writes a self-contained HTML report.
All figures are embedded as base64 PNG -- the output file is fully portable.

Usage:
    python generate_report.py
Output:
    orojo_screening_report.html

# STYLE_MODE:
# "clean"  = baseline technical report
# "styled" = presentation-enhanced risk report preview
STYLE_MODE = "clean"
"""

import base64, io, os, sys, math, zipfile, tempfile
from datetime import date

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping, box
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from rasterio.plot import show as rshow
from rasterio.warp import reproject as warp_reproject, Resampling, calculate_default_transform as calc_transform
import geodatasets
import cartopy.io.shapereader as shpreader

sys.path.insert(0, r'C:\Users\sayon\Orojo')
from screening.config import DATASETS
from screening.load_boundary import load_boundary_file
from screening.t1_bbox import normalise_bbox
from screening.t2_resolve import ecological_identity
from screening.t3_estoque import land_cover_history, _read_vat_from_dbf
from screening.t4_griscom import restoration_opportunity, _read_vat
from screening.t5_albedo import climate_implications
from screening.t6_iplc import run as _t6_run
from screening import t3b_glc_fcs as _t3b_glc_fcs

# ---------------------------------------------------------------------------
# Project settings
# ---------------------------------------------------------------------------
OROJO_KML      = r'C:\Users\sayon\Orojo\Orojo Forestry Lands.kml'
PROJECT_ID     = 'ma_tatou_nz'
PROJECT_LABEL  = 'Ma Tatou Restoration Sites, Canterbury, South Island, NZ'
STYLE_MODE = "clean"   # "clean" or "styled"  —  toggle presentation enhancements
# BOUNDARY_PATH: set to OROJO_KML for Orojo, or a .geojson path for other projects.
# .geojson files are loaded directly via gpd.read_file(); all other formats use load_boundary_file().
BOUNDARY_PATH  = r'C:\Users\sayon\Orojo\ma_tatou_canterbury.geojson'

# Restoration-context note: appends a caveat to the T3b section for projects where
# forest increase is the intended outcome, not a screening concern.
# Does not affect flags, thresholds, or any other section.
RESTORATION_CONTEXT_NOTE = (PROJECT_ID == 'ma_tatou_nz')

OUTPUT_PATH = (
    r'C:\Users\sayon\Orojo\ma_tatou_nz_screening_report_preview.html'
    if STYLE_MODE == "styled" else
    r'C:\Users\sayon\Orojo\ma_tatou_nz_screening_report.html'
)

T2_FIG1_PATH   = 'fig1_reference_domains.png'
T2_FIG2_PATH   = 'fig2_domain_b_breakdown.png'
T6_FIG_PATH    = 'orojo_iplc_proximity_map.png'
T3B_CSV_PATH   = f'{PROJECT_ID}_glc_timeseries.csv'
T3B_FIG_PATH   = f'{PROJECT_ID}_glc_timeseries.png'
IPLC_DATA_DIR  = r'C:\Users\sayon\Orojo\IPLC Landmark Data'

# Comparability ratings for Domain B ecoregions (mirrors fig2_domain_b_breakdown.py)
_T2_COMPARABILITY = {
    'Llanos':                                  'HIGH',
    'Beni savanna':                            'HIGH',
    'Guianan savanna':                         'HIGH',
    'Cerrado':                                 'MEDIUM',
    'Humid Chaco':                             'MEDIUM',
    'Dry Chaco':                               'LOW',
    'Uruguayan savanna':                       'LOW',
    'Campos Rupestres montane savanna':        'LOW',
    'Miskito pine forests':                    'LOW',
    'Belizian pine savannas':                  'LOW',
    'Clipperton Island shrub and grasslands':  'LOW',
}
_T2_BADGE_CLS = {
    'HIGH':   'comp-badge-high',
    'MEDIUM': 'comp-badge-medium',
    'LOW':    'comp-badge-low',
}

def _fmt_area(km2):
    """Format km² values as human-readable strings (M, k, or raw)."""
    if km2 >= 1_000_000:
        return f'{km2 / 1_000_000:.1f}M km&#178;'
    elif km2 >= 1_000:
        return f'{round(km2 / 1_000):.0f}k km&#178;'
    else:
        return f'{km2:,.0f} km&#178;'

# Rendering helpers — conditioned on STYLE_MODE; used throughout HTML assembly
_figure_wrap_open   = '<div class="figure-block">' if STYLE_MODE == "styled" else ''
_figure_wrap_close  = '</div>'                       if STYLE_MODE == "styled" else ''
_method_cls         = ' method-table'                if STYLE_MODE == "styled" else ''
# CSS value helpers
_body_line_height   = '1.55' if STYLE_MODE == "styled" else '1.65'
_h2_border_width    = '5px'  if STYLE_MODE == "styled" else '4px'
_ctx_bg             = '#f4f4f4' if STYLE_MODE == "styled" else '#fffbf0'
_ctx_border         = '#9e9e9e' if STYLE_MODE == "styled" else '#e6a817'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def png_to_b64(path):
    """Read a pre-generated PNG from disk and return base64, or None if missing."""
    if not os.path.isfile(path):
        print(f'  [WARNING] Figure not found, skipping in report: {path}')
        return None
    with open(path, 'rb') as fh:
        return base64.b64encode(fh.read()).decode('utf-8')

def badge(text, colour):
    return f'<span class="badge badge-{colour}">{text}</span>'

# ── Shared map settings ───────────────────────────────────────────────────────
# MAP_XLIM / MAP_YLIM / _CY_SITE derived from T1 after boundary loaded (below)
SITE_PAD      = 0.15   # degrees of padding on every side of the project boundary
SITE_SCALE_KM = 5      # scale bar length for site maps (T2–T5); small for actual parcel

def _north_arrow(ax):
    """North arrow in upper-right corner (axes fraction coords)."""
    ax.annotate('N', xy=(0.965, 0.175), xycoords='axes fraction',
                fontsize=12, fontweight='bold', ha='center', fontfamily='Arial',
                color='#333', zorder=8)
    ax.annotate('', xy=(0.965, 0.215), xycoords='axes fraction',
                xytext=(0.965, 0.13), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.8), zorder=8)

def _scale_bar(ax, km=100):
    """Scale bar at lower-right of current axes extent (must call after set_xlim/ylim)."""
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    lat  = (ylim[0] + ylim[1]) / 2
    deg  = km / (math.cos(math.radians(lat)) * 111.32)
    xw   = xlim[1] - xlim[0];  yw = ylim[1] - ylim[0]
    x0   = xlim[1] - 0.03 * xw - deg
    y0   = ylim[0] + 0.04 * yw
    ax.plot([x0, x0 + deg], [y0, y0], color='#333', lw=2.5,
            solid_capstyle='butt', zorder=8, clip_on=False)
    for tx in [x0, x0 + deg]:
        ax.plot([tx, tx], [y0 - 0.008 * yw, y0 + 0.008 * yw],
                color='#333', lw=1.5, zorder=8, clip_on=False)
    ax.text(x0 + deg / 2, y0 - 0.025 * yw, f'{km} km',
            ha='center', fontsize=8, color='#333', fontfamily='Arial',
            zorder=8, clip_on=False)

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
print('Loading project boundary...')
if BOUNDARY_PATH.lower().endswith('.geojson'):
    print(f'  [load_boundary] Loading GeoJSON directly: {BOUNDARY_PATH}')
    boundary_gdf = gpd.read_file(BOUNDARY_PATH)
else:
    boundary_gdf = load_boundary_file(BOUNDARY_PATH)

print('Running T1...')
t1 = normalise_bbox(boundary_gdf)
bbox_gdf = t1['bbox_gdf']

# ── Derive shared map extent from actual project boundary ─────────────────────
_bb      = t1['bounds_wgs84']
MAP_XLIM = (_bb[0] - SITE_PAD, _bb[2] + SITE_PAD)
MAP_YLIM = (_bb[1] - SITE_PAD, _bb[3] + SITE_PAD)
_CY_SITE = (_bb[1] + _bb[3]) / 2

print('Running T2...')
t2 = ecological_identity(bbox_gdf)

print('Running T3...')
t3 = land_cover_history(bbox_gdf)

print('Running T3b...')
t3b = _t3b_glc_fcs.run(T3B_CSV_PATH, biome_name=t2['dominant_biome'])

print('Running T4...')
t4 = restoration_opportunity(bbox_gdf, dominant_biome=t2['dominant_biome'])

print('Running T5...')
t5 = climate_implications(bbox_gdf)

print('Running T6...')
_t6_geom = boundary_gdf.to_crs('EPSG:4326').geometry.iloc[0]
if _t6_geom.geom_type == 'MultiPolygon':
    _t6_geom = max(_t6_geom.geoms, key=lambda g: g.area)
_t6_coords = [list(c) for c in _t6_geom.exterior.coords]
t6 = _t6_run(_t6_coords, IPLC_DATA_DIR)

print('Generating figures...')

# ---------------------------------------------------------------------------
# Figure 1: Two-panel regional context map
#   Left  — South America locator (site position in regional context)
#   Right — Colombia with department boundaries, bbox, north arrow, scale bar
# ---------------------------------------------------------------------------
admin0_path = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
admin1_path = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
admin0_gdf  = gpd.read_file(admin0_path)
admin1_gdf  = gpd.read_file(admin1_path)

col_depts   = admin1_gdf[admin1_gdf['iso_a2'] == 'CO']
colombia    = admin0_gdf[admin0_gdf['ISO_A2'] == 'CO']
# Neighbours visible in the Colombia panel
neighbours  = admin0_gdf[admin0_gdf['ISO_A2'].isin(['VE', 'BR', 'PE', 'EC', 'PA', 'GY', 'SR'])]
# South America clip for locator (broad bbox)
sa_bbox     = box(-82, -58, -30, 15)
south_am    = admin0_gdf[admin0_gdf.geometry.intersects(sa_bbox)]

minx, miny, maxx, maxy = t1['bounds_wgs84']
cx, cy = t1['centroid']
km_per_deg_lon = math.cos(math.radians(cy)) * 111.32   # for scale bar

fig, (ax_loc, ax_main) = plt.subplots(
    1, 2, figsize=(13, 7),
    gridspec_kw={'width_ratios': [1, 2.3]}
)

# ── Left panel: South America locator ─────────────────────────────────────────
ax_loc.set_facecolor('#d6eaf8')
south_am.plot(ax=ax_loc, color='#e8f0e8', edgecolor='#aaa', linewidth=0.35)
colombia.plot(ax=ax_loc, color='#b8d4b8', edgecolor='#555', linewidth=0.9)
# bbox shown as filled red rectangle
from matplotlib.patches import Rectangle as MPLRect
rect_loc = MPLRect(
    (minx, miny), maxx - minx, maxy - miny,
    linewidth=1.8, edgecolor='#c0392b', facecolor='#e74c3c55', zorder=5
)
ax_loc.add_patch(rect_loc)
ax_loc.set_xlim(-82, -32)
ax_loc.set_ylim(-57, 15)
ax_loc.set_xticks([])
ax_loc.set_yticks([])
ax_loc.set_title('Regional context', fontsize=9, fontweight='bold',
                 fontfamily='Arial', pad=5)
for sp in ax_loc.spines.values():
    sp.set_edgecolor('#888')

# ── Right panel: Colombia + departments ────────────────────────────────────────
ax_main.set_facecolor('#d6eaf8')
neighbours.plot(ax=ax_main, color='#ececec', edgecolor='#aaa', linewidth=0.5)
colombia.plot(ax=ax_main, color='#f2f8f2', edgecolor='#555', linewidth=1.2)
col_depts.plot(ax=ax_main, color='none', edgecolor='#b0b0b0',
               linewidth=0.45, linestyle='--')
bbox_gdf.plot(ax=ax_main, facecolor='#e74c3c18', edgecolor='#c0392b', linewidth=2.5)
ax_main.plot(cx, cy, marker='+', color='#c0392b', markersize=13, markeredgewidth=2, zorder=6)
ax_main.annotate(
    f'Project area\n(Vichada, Colombia)\n{t1["area_km2"]:,.0f} km\u00b2',
    xy=(cx, cy), xytext=(cx + 1.6, cy + 1.4),
    fontsize=8.5, color='#c0392b', fontfamily='Arial',
    arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2), zorder=7
)

# Country name labels
for _, row in neighbours.iterrows():
    pt = row.geometry.centroid
    if -80 <= pt.x <= -65.5 and -4.5 <= pt.y <= 13.5:
        ax_main.text(pt.x, pt.y, row['SOVEREIGNT'], fontsize=7, color='#888',
                     ha='center', va='center', fontfamily='Arial',
                     style='italic', zorder=4)

ax_main.set_xlim(-80, -65.5)
ax_main.set_ylim(-4.5, 13.5)
ax_main.set_xlabel('Longitude', fontsize=9, fontfamily='Arial')
ax_main.set_ylabel('Latitude', fontsize=9, fontfamily='Arial')
ax_main.set_title(f'{PROJECT_LABEL}', fontsize=11, fontweight='bold', pad=10)
ax_main.grid(True, alpha=0.18, linestyle=':')
ax_main.tick_params(labelsize=8)

# North arrow (axes fraction coords)
ax_main.annotate('N', xy=(0.965, 0.175), xycoords='axes fraction',
                 fontsize=12, fontweight='bold', ha='center', fontfamily='Arial',
                 color='#333', zorder=8)
ax_main.annotate('', xy=(0.965, 0.215), xycoords='axes fraction',
                 xytext=(0.965, 0.13), textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='#333', lw=1.8), zorder=8)

# Scale bar: 200 km at bottom right in data coords
scale_km   = 200
scale_deg  = scale_km / km_per_deg_lon
scale_x0   = -67.8
scale_y    = -3.8
ax_main.plot([scale_x0, scale_x0 + scale_deg], [scale_y, scale_y],
             color='#333', linewidth=2.5, solid_capstyle='butt', zorder=8)
for tick_x in [scale_x0, scale_x0 + scale_deg]:
    ax_main.plot([tick_x, tick_x], [scale_y - 0.12, scale_y + 0.12],
                 color='#333', lw=1.5, zorder=8)
ax_main.text(scale_x0 + scale_deg / 2, scale_y - 0.38, f'{scale_km} km',
             ha='center', fontsize=8, color='#333', fontfamily='Arial', zorder=8)

plt.tight_layout()
img_map = fig_to_b64(fig)

# ---------------------------------------------------------------------------
# Figure 2: RESOLVE ecoregion map — official biome colour palette + pie inset
# ---------------------------------------------------------------------------
resolve_raw = gpd.read_file(DATASETS['resolve'])
bx0, by0, bx1, by1 = bbox_gdf.total_bounds
PAD2 = 1.5   # degree padding around bbox for map extent
candidates = resolve_raw.cx[bx0 - PAD2:bx1 + PAD2, by0 - PAD2:by1 + PAD2]
clipped_eco = gpd.overlay(
    resolve_raw.cx[bx0:bx1, by0:by1],
    bbox_gdf[['geometry']], how='intersection'
)

# Official RESOLVE biome colours from COLOR_BIO field
# Each ecoregion inherits its biome hex; ecoregions within same biome share a colour
eco_bio_color = dict(zip(clipped_eco['ECO_NAME'], clipped_eco['COLOR_BIO']))
eco_names = clipped_eco['ECO_NAME'].unique()

# Biome summary for pie inset — use t2 ecoregion_table (has pct), join COLOR_BIO
biome_color_lookup = (
    resolve_raw[['BIOME_NAME', 'COLOR_BIO']]
    .drop_duplicates('BIOME_NAME')
    .set_index('BIOME_NAME')['COLOR_BIO']
    .to_dict()
)
biome_pct_series = t2['ecoregion_table'].groupby('BIOME_NAME')['pct'].sum()
bio_pct = (
    biome_pct_series
    .reset_index()
    .rename(columns={'pct': 'pct'})
    .assign(COLOR_BIO=lambda df: df['BIOME_NAME'].map(biome_color_lookup))
    .sort_values('pct', ascending=False)
    .reset_index(drop=True)
)

# ── Main map ─────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 6.5))

# Background context: ecoregions just outside bbox (candidates not in clipped)
# Plot unclipped candidates at low alpha so site is clearly focal
candidates.plot(ax=ax2, color='#ececec', edgecolor='#d0d0d0', linewidth=0.2, alpha=0.6)

# Clipped ecoregions in official biome colours
for eco in eco_names:
    subset = clipped_eco[clipped_eco['ECO_NAME'] == eco]
    color  = eco_bio_color.get(eco, '#cccccc')
    subset.plot(ax=ax2, color=color, edgecolor='white', linewidth=0.4, alpha=0.92)

# Project bbox boundary (same style as T1)
bbox_gdf.plot(ax=ax2, facecolor='none', edgecolor='#c0392b', linewidth=2.5)

# ── Legend (ecoregions, labelled with biome name for clarity) ─────────────
# Group patches by biome so legend isn't redundant
biome_legend_seen = set()
legend_patches = []
for _, row in bio_pct.iterrows():
    biome = row['BIOME_NAME']
    color = row['COLOR_BIO']
    pct   = row['pct']
    label = f'{biome} ({pct:.1f}%)'
    if biome not in biome_legend_seen:
        legend_patches.append(mpatches.Patch(color=color, label=label))
        biome_legend_seen.add(biome)

ax2.legend(handles=legend_patches, fontsize=7.5, loc='upper center',
           bbox_to_anchor=(0.5, -0.10), ncol=2,
           framealpha=0.93, title='RESOLVE Biome', title_fontsize=8,
           edgecolor='#bbb')

# Map extent: bbox + padding
ax2.set_xlim(*MAP_XLIM)
ax2.set_ylim(*MAP_YLIM)
ax2.set_xlabel('Longitude', fontsize=9, fontfamily='Arial')
ax2.set_ylabel('Latitude', fontsize=9, fontfamily='Arial')
ax2.set_title('RESOLVE Ecoregions — Project Area and Surrounding Context',
              fontsize=11, fontweight='bold', pad=10, fontfamily='Arial')
ax2.grid(True, alpha=0.18, linestyle=':')
ax2.tick_params(labelsize=8)
_north_arrow(ax2)
_scale_bar(ax2, km=SITE_SCALE_KM)

plt.tight_layout()
fig2.subplots_adjust(bottom=0.22)   # room for legend below axes

# ── Pie chart inset (biome % composition) — added after tight_layout ────────
ax_pie = fig2.add_axes([0.68, 0.60, 0.27, 0.27])   # upper-right corner
pie_labels = [b[:22] + '…' if len(b) > 22 else b for b in bio_pct['BIOME_NAME']]
ax_pie.pie(
    bio_pct['pct'], labels=None,
    colors=bio_pct['COLOR_BIO'].tolist(),
    startangle=90, counterclock=False,
    wedgeprops=dict(edgecolor='white', linewidth=0.8)
)
ax_pie.set_title('Biome mix', fontsize=7.5, fontfamily='Arial', pad=3)

img_eco = fig_to_b64(fig2)
print('  Loading pre-generated T2 reference domain figures...')
img_ref_domains = png_to_b64(T2_FIG1_PATH) if PROJECT_ID == 'orojo_colombia' else None
img_domain_b    = png_to_b64(T2_FIG2_PATH) if PROJECT_ID == 'orojo_colombia' else None

# ---------------------------------------------------------------------------
# Figure 3: Forest cover trajectory — spatial choropleth + bar chart (2-panel)
# ---------------------------------------------------------------------------
cat_colors = {
    'Persistent forest':          '#1a6b2a',
    'Persistent non-forest':      '#c8a832',
    'Non-persistent forest loss': '#c0392b',
    'Non-persistent forest gain': '#27ae60',
    'Persistent forest loss':     '#8b0000',
    'Non-persistent forest':      '#5aad6e',
    'Non-persistent non-forest':  '#d4c97a',
    'Persistent forest gain':     '#52c26a',
}

def _hex_to_rgb01(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

cat_rgb = {cat: _hex_to_rgb01(col) for cat, col in cat_colors.items()}

# Re-open Estoque raster — clip in native Eckert IV, then reproject to WGS84 for display.
# Statistics are unchanged (computed from native raster in t3 above).
# Nearest-neighbor resampling is the only valid method for categorical pixel values.
with zipfile.ZipFile(DATASETS['estoque']) as zf3:
    tif3_name = next(n for n in zf3.namelist() if n.endswith('.tif') and not n.endswith('.ovr'))
    dbf3_name = next(n for n in zf3.namelist() if n.endswith('.vat.dbf'))
    vat3 = _read_vat_from_dbf(zf3.read(dbf3_name))
    with tempfile.TemporaryDirectory() as tmpdir3:
        zf3.extract(tif3_name, tmpdir3)
        tif3_path = os.path.join(tmpdir3, tif3_name)
        with rasterio.open(tif3_path) as src3:
            raster_crs3 = src3.crs
            nodata3     = int(src3.nodata) if src3.nodata is not None else 255
            bbox_r3     = bbox_gdf.to_crs(raster_crs3)
            geoms3      = [mapping(g) for g in bbox_r3.geometry]
            cl3, tr3    = mask(src3, geoms3, crop=True, filled=False)
            data3       = cl3[0]
            nrows3, ncols3 = data3.shape

            # Reproject clipped array → WGS84 for display
            dst_crs3 = rasterio.crs.CRS.from_epsg(4326)
            bounds3  = rasterio.transform.array_bounds(nrows3, ncols3, tr3)
            tr3_wgs, w_wgs, h_wgs = calc_transform(
                raster_crs3, dst_crs3, ncols3, nrows3,
                left=bounds3[0], bottom=bounds3[1], right=bounds3[2], top=bounds3[3]
            )
            src_arr3 = data3.data.astype(np.uint8) if hasattr(data3, 'data') else data3.astype(np.uint8)
            if hasattr(data3, 'mask'):
                src_arr3[data3.mask] = nodata3
            data3_wgs = np.full((h_wgs, w_wgs), nodata3, dtype=np.uint8)
            warp_reproject(
                source=src_arr3,
                destination=data3_wgs,
                src_transform=tr3,
                src_crs=raster_crs3,
                dst_transform=tr3_wgs,
                dst_crs=dst_crs3,
                resampling=Resampling.nearest,
                src_nodata=nodata3,
                dst_nodata=nodata3,
            )

# Build RGBA array from WGS84-reprojected data
valid_wgs = (data3_wgs != nodata3)
rgba3 = np.zeros((h_wgs, w_wgs, 4), dtype=float)
for pv in np.unique(data3_wgs[valid_wgs]).tolist():
    label3 = vat3.get(int(pv))
    if label3 and label3 in cat_rgb:
        px = valid_wgs & (data3_wgs == pv)
        rgba3[px, :3] = cat_rgb[label3]
        rgba3[px, 3]  = 1.0

# Extent in WGS84 degrees — bbox is a perfect rectangle here, no background bleed
left3_w   = tr3_wgs.c
right3_w  = tr3_wgs.c + tr3_wgs.a * w_wgs
top3_w    = tr3_wgs.f
bottom3_w = tr3_wgs.f + tr3_wgs.e * h_wgs

# ── Figure: full-width spatial choropleth (WGS84, no clip_path needed) ────────
cats   = sorted(t3['category_pct'].items(), key=lambda x: -x[1])
labels = [c[0] for c in cats]
values = [c[1] for c in cats]

fig3, ax_map3 = plt.subplots(figsize=(10, 7.5))
im3 = ax_map3.imshow(rgba3, extent=[left3_w, right3_w, bottom3_w, top3_w],
                     origin='upper', aspect='auto')
bbox_gdf.boundary.plot(ax=ax_map3, edgecolor='#c0392b', linewidth=2)

ax_map3.set_title('Forest Cover Trajectories 1960–2019 (HILDA+ via Estoque et al. 2022)',
                  fontsize=11, fontweight='bold', fontfamily='Arial', pad=10)
ax_map3.set_xlabel('Longitude', fontsize=9)
ax_map3.set_ylabel('Latitude', fontsize=9)
ax_map3.tick_params(labelsize=8)
ax_map3.grid(True, alpha=0.18, linestyle=':')
ax_map3.set_xlim(*MAP_XLIM)
ax_map3.set_ylim(*MAP_YLIM)
_north_arrow(ax_map3)
_scale_bar(ax_map3, km=SITE_SCALE_KM)

lc_patches = [mpatches.Patch(color=cat_colors.get(l, '#999'), label=f'{l} ({v:.1f}%)')
              for l, v in zip(labels, values) if v > 0]
ax_map3.legend(handles=lc_patches, fontsize=8, loc='upper center',
               bbox_to_anchor=(0.5, -0.10), ncol=3,
               framealpha=0.93, title='Trajectory category', title_fontsize=8.5,
               edgecolor='#bbb')

plt.tight_layout()
fig3.subplots_adjust(bottom=0.20)
img_lc = fig_to_b64(fig3)


# ---------------------------------------------------------------------------
# Figure 4: NCS map — WGS84 display, T3 persistent forest underlay
# ---------------------------------------------------------------------------
# Reproject Griscom raster to WGS84 (nearest-neighbor; binary categorical data)
vat_g4 = _read_vat(DATASETS['griscom'] + '.vat.dbf')
with rasterio.open(DATASETS['griscom']) as src:
    crs_g4     = src.crs
    bbox_g4    = bbox_gdf.to_crs(crs_g4)
    cl_g4, tr_g4 = mask(src, [mapping(g) for g in bbox_g4.geometry], crop=True, filled=False)
    data_g4    = cl_g4[0]

nrows_g4, ncols_g4 = data_g4.shape
raw_g4  = data_g4.data.astype(np.int64) if hasattr(data_g4, 'data') else data_g4.astype(np.int64)
valid_g4 = ~data_g4.mask if hasattr(data_g4, 'mask') else np.ones(raw_g4.shape, bool)

# Encode to uint8: 0=not NCS, 1=NCS, 255=nodata
ncs_u8 = np.full(raw_g4.shape, 255, dtype=np.uint8)
for pv4, flag4 in vat_g4.items():
    ncs_u8[valid_g4 & (raw_g4 == pv4)] = int(flag4)

bounds_g4 = rasterio.transform.array_bounds(nrows_g4, ncols_g4, tr_g4)
_wgs84    = rasterio.crs.CRS.from_epsg(4326)
tr4w, w4w, h4w = calc_transform(crs_g4, _wgs84, ncols_g4, nrows_g4,
                                 left=bounds_g4[0], bottom=bounds_g4[1],
                                 right=bounds_g4[2], top=bounds_g4[3])
data4w = np.full((h4w, w4w), 255, dtype=np.uint8)
warp_reproject(source=ncs_u8, destination=data4w,
               src_transform=tr_g4, src_crs=crs_g4,
               dst_transform=tr4w, dst_crs=_wgs84,
               resampling=Resampling.nearest, src_nodata=255, dst_nodata=255)

l4w = tr4w.c;  r4w = tr4w.c + tr4w.a * w4w
t4w = tr4w.f;  b4w = tr4w.f + tr4w.e * h4w

rgba4 = np.zeros((h4w, w4w, 4), dtype=float)
rgba4[data4w == 0] = [0.80, 0.80, 0.80, 1.0]   # grey: not eligible
rgba4[data4w == 1] = [0.18, 0.49, 0.20, 1.0]   # green: NCS eligible

fig4, ax4 = plt.subplots(figsize=(9, 7.5))
ax4.imshow(rgba4, extent=[l4w, r4w, b4w, t4w], origin='upper', aspect='auto')
bbox_gdf.boundary.plot(ax=ax4, edgecolor='#c0392b', linewidth=2)
ax4.set_xlim(*MAP_XLIM)
ax4.set_ylim(*MAP_YLIM)
ax4.set_title('Griscom NCS Reforestation Opportunity (decoded via VAT)',
              fontsize=11, fontweight='bold', pad=10, fontfamily='Arial')
ax4.set_xlabel('Longitude', fontsize=9)
ax4.set_ylabel('Latitude', fontsize=9)
ax4.tick_params(labelsize=8)
ax4.grid(True, alpha=0.18, linestyle=':')
_north_arrow(ax4)
_scale_bar(ax4, km=SITE_SCALE_KM)
p4 = [
    mpatches.Patch(color='#2e7d32', label=f'NCS reforestation opportunity ({t4["ncs_overlap_pct"]:.1f}%)'),
    mpatches.Patch(color='#cccccc', label=f'Not eligible per Griscom ({100 - t4["ncs_overlap_pct"]:.1f}%)'),
]
ax4.legend(handles=p4, fontsize=9, loc='upper center',
           bbox_to_anchor=(0.5, -0.08), ncol=2, framealpha=0.9, edgecolor='#bbb')
plt.tight_layout()
fig4.subplots_adjust(bottom=0.14)
img_ncs = fig_to_b64(fig4)

# ---------------------------------------------------------------------------
# Figure 5: Albedo spatial map + histogram + bucket bar (3 panels)
# ---------------------------------------------------------------------------
with rasterio.open(DATASETS['albedo']) as src:
    g5      = [mapping(g) for g in bbox_gdf.geometry]
    cl5, tr5 = mask(src, g5, crop=True, filled=False)
    data5   = cl5[0]

# Build spatial display array: mask nodata and cap pixels as NaN
alb_map_arr = data5.data.astype(float)
if hasattr(data5, 'mask'):
    alb_map_arr[data5.mask] = np.nan
alb_map_arr[np.abs(alb_map_arr) >= 10_000] = np.nan

# Affine extent for imshow (EPSG:4326 matches bbox_gdf CRS)
left5   = tr5.c
right5  = tr5.c + tr5.a * alb_map_arr.shape[1]
top5    = tr5.f
bottom5 = tr5.f + tr5.e * alb_map_arr.shape[0]   # tr5.e is negative

# Diverging blue–white–red palette: blue = cooling (+), white = 0, red = warming (-)
cmap_div = plt.cm.RdBu.copy()
cmap_div.set_bad('lightgray', alpha=0.5)
VMAX_DISPLAY = 100   # symmetric display range in percent

# Layout: map (same footprint as T2–T4) on top; Hasler category bar below
fig5 = plt.figure(figsize=(10, 10))
gs5  = fig5.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=0.38)
ax_map5 = fig5.add_subplot(gs5[0])
ax_bar5 = fig5.add_subplot(gs5[1])

# --- Spatial map ---
im5 = ax_map5.imshow(
    alb_map_arr, cmap=cmap_div, vmin=-VMAX_DISPLAY, vmax=VMAX_DISPLAY,
    extent=[left5, right5, bottom5, top5], origin='upper', aspect='auto'
)
bbox_gdf.boundary.plot(ax=ax_map5, edgecolor='#c0392b', linewidth=2)

# Colorbar with Hasler breakpoints
cb5 = fig5.colorbar(im5, ax=ax_map5,
                    label='Albedo offset (% of carbon sequestration benefit)',
                    shrink=0.65, pad=0.02, fraction=0.025)
cb5.set_ticks([-100, -50, 0, 50, 100])
cb5.set_ticklabels(['-100%\n(net-negative)', '-50%\n(substantial)', '0%', '+50%', '+100%'])
cb5.ax.tick_params(labelsize=7.5)

ax_map5.set_title('Hasler et al. 2024 — Albedo Offset from Reforestation',
                  fontsize=11, fontweight='bold', fontfamily='Arial', pad=10)
ax_map5.set_xlabel('Longitude', fontsize=9)
ax_map5.set_ylabel('Latitude', fontsize=9)
ax_map5.grid(True, alpha=0.2, linestyle=':')
ax_map5.set_xlim(*MAP_XLIM)
ax_map5.set_ylim(*MAP_YLIM)
_north_arrow(ax_map5)
_scale_bar(ax_map5, km=SITE_SCALE_KM)

# --- Hasler category bar chart ---
warming_mid = max(0, t5['pct_negative'] - t5['pct_substantial'])
buckets5 = {
    'Cooling\n(> 0%)':                t5['pct_positive'],
    'Mild warming\n(0 to −50%)':       warming_mid,
    'Substantial\n(−50 to −100%)':     max(0, t5['pct_substantial'] - t5['pct_net_negative']),
    'Net climate-negative\n(< −100%)': t5['pct_net_negative'],
}
cols5 = ['#2196F3', '#FF9800', '#F44336', '#880000']
bars5 = ax_bar5.bar(range(len(buckets5)), list(buckets5.values()),
                    color=cols5, edgecolor='white', linewidth=0.5, width=0.5)
ax_bar5.set_xticks(range(len(buckets5)))
ax_bar5.set_xticklabels(list(buckets5.keys()), fontsize=9)
ax_bar5.set_ylabel('% of project area', fontsize=9)
ax_bar5.set_title(
    f'Albedo Offset by Hasler Category  |  Median: {t5["median_offset_pct"]:+.1f}%',
    fontsize=10, fontweight='bold')
ax_bar5.spines['top'].set_visible(False)
ax_bar5.spines['right'].set_visible(False)
for b, v in zip(bars5, buckets5.values()):
    ax_bar5.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                 f'{v:.1f}%', ha='center', fontsize=9.5, fontweight='bold')

img_alb    = fig_to_b64(fig5)
img_t6_map = png_to_b64(T6_FIG_PATH) if PROJECT_ID == 'orojo_colombia' else None
img_t3b    = png_to_b64(T3B_FIG_PATH) if t3b else None

# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------
today = date.today().strftime('%d %B %Y')

_t3_pnf_exec  = t3['category_pct'].get('Persistent non-forest', 0)

_t3b_flag_color = {'RED': '#c62828', 'AMBER': '#e65100', 'GREEN': '#1b5e20'}.get(
    t3b['flag'] if t3b else 'GREEN', '#555'
)
_t3b_exec_row = (
    f'    <div>\n'
    f'      <div class="sg-label">GLC_FCS30D (30m, mean 1985-2022)</div>\n'
    f'      <div class="sg-value" style="font-size:0.9em">'
    f'{t3b["mean_grassland_pct"]:.1f}% grassland / '
    f'{t3b["mean_forest_pct_post2000"]:.1f}% forest &nbsp;'
    f'<span style="color:{_t3b_flag_color}">{t3b["flag"]}</span></div>\n'
    f'    </div>'
    if t3b else ''
)

eco_rows = ''.join(
    f'<tr><td>{r["ECO_NAME"]}</td><td>{r["BIOME_NAME"]}</td>'
    f'<td>{r["area_km2"]:,.0f}</td><td>{r["pct"]:.1f}%</td></tr>'
    for _, r in t2['ecoregion_table'].iterrows()
)

lc_rows = ''.join(
    f'<tr><td>{cat}</td><td>{pct:.1f}%</td>'
    f'<td><div class="minibar" style="width:{max(int(pct),1)}%;'
    f'background:{cat_colors.get(cat,"#999")}"></div></td></tr>'
    for cat, pct in sorted(t3['category_pct'].items(), key=lambda x: -x[1])
)

_t6_flag_entry = (
    f'POSITIVE: T6 IPLC: no features within {t6["audit"]["buffer_km"]:.0f} km proximity buffer'
    if t6['flag'] == 'GREEN' else
    f'AMBER: T6 IPLC: {len(t6["within_buf"])} {"territory" if len(t6["within_buf"]) == 1 else "territories"} within {t6["audit"]["buffer_km"]:.0f} km'
    if t6['flag'] == 'AMBER' else
    f'RED: T6 IPLC: direct overlap with {len(t6["overlap"])} {"territory" if len(t6["overlap"]) == 1 else "territories"}, FPIC triggered'
)
all_flags = t2['flags'] + t3['flags'] + (t3b['flags'] if t3b else []) + t4['flags'] + t5['flags'] + [_t6_flag_entry]
flag_rows = ''
for f in all_flags:
    if f.startswith('POSITIVE'):
        cls = 'flag-positive'
    elif f.startswith('RED') or f.startswith('FLAG'):
        cls = 'flag-red'
    elif f.startswith('CONTEXT'):
        cls = 'flag-note'
    else:
        cls = 'flag-note'
    flag_rows += f'<tr class="{cls}"><td>{f}</td></tr>'

# Summary badges
if t4['signal'] == 'FLAG':
    t4_badge = badge('FLAG (expected biome)', 'orange') if t4['grassland_context'] else badge('FLAG', 'red')
elif t4['signal'] == 'PARTIAL':
    t4_badge = badge('PARTIAL', 'orange')
else:
    t4_badge = badge('POSITIVE', 'green')

t5_badge = badge('POSITIVE', 'green') if t5['median_offset_pct'] > 0 else badge('FLAG', 'red')

# T4 guardrail badge
if t4['signal'] == 'FLAG' and t4['grassland_context']:
    t4_guardrail_badge = badge('FLAG (expected)', 'orange')
elif t4['signal'] == 'FLAG':
    t4_guardrail_badge = badge('FLAG', 'red')
else:
    t4_guardrail_badge = badge(t4['signal'], 'green')

t4_context_note = (
    f"Expected outcome: dominant biome ({t2['dominant_biome']}) is excluded by the "
    f"Griscom safeguard. Low NCS overlap reflects the dataset design, not a data problem."
    if t4['grassland_context'] else ""
)

# T2 reference domain figures — inserted between img_eco and comment-box
_t2_dominant_eco = (
    t2['ecoregion_table']['ECO_NAME'].iloc[0]
    if len(t2['ecoregion_table']) > 0 else 'Llanos'
)
_eco_label = f'{_t2_dominant_eco} savanna' if PROJECT_ID == 'orojo_colombia' else _t2_dominant_eco

if t6['flag'] == 'RED':
    _exec_narrative = (
        f'Native {_eco_label} site ({_t3_pnf_exec:.0f}% persistent non-forest, 1960-2019). '
        + (f'Primary finding: the parcel directly overlaps a mapped Resguardo Indigena, '
           f'triggering FPIC under ILO Convention 169 / Colombian Law 21/1991 (T6 RED).'
           if PROJECT_ID == 'orojo_colombia' else
           f'Primary finding: the parcel directly overlaps a mapped IPLC territory, '
           f'triggering FPIC requirements (T6 RED).')
    )
elif t6['flag'] == 'AMBER':
    _exec_narrative = (
        f'Native {_eco_label} site ({_t3_pnf_exec:.0f}% persistent non-forest, 1960-2019). '
        f'{len(t6["within_buf"])} IPLC {"territory" if len(t6["within_buf"]) == 1 else "territories"} '
        f'within {t6["audit"]["buffer_km"]:.0f} km warrant stakeholder assessment before project financing (T6 AMBER).'
        if PROJECT_ID == 'orojo_colombia' else
        f'Restoration baseline: {_eco_label} ({_t3_pnf_exec:.0f}% persistent non-forest, 1960-2019). '
        f'{len(t6["within_buf"])} IPLC {"territory" if len(t6["within_buf"]) == 1 else "territories"} '
        f'within {t6["audit"]["buffer_km"]:.0f} km reflect indigenous land tenure context of the project area (T6 AMBER).'
    )
else:
    _exec_narrative = (
        f'Native {_eco_label} site ({_t3_pnf_exec:.0f}% persistent non-forest, 1960-2019). '
        f'No IPLC features within {t6["audit"]["buffer_km"]:.0f} km (T6 GREEN). '
        f'Albedo screening indicates a net cooling co-benefit from vegetation establishment.'
    )

if img_ref_domains and img_domain_b:
    # Compute Domain B ecoregion table (same filtering as fig2_domain_b_breakdown.py)
    print('  Computing Domain B ecoregion ranking table...')
    _resolve_b = gpd.read_file(DATASETS['resolve'])
    _resolve_b = _resolve_b[_resolve_b['REALM'] == t2['dominant_realm']]
    _resolve_b = _resolve_b[_resolve_b['BIOME_NAME'] == t2['dominant_biome']]
    _resolve_b = _resolve_b[['ECO_NAME', 'geometry']].copy()
    _resolve_b['area_km2'] = _resolve_b.to_crs('EPSG:6933').geometry.area / 1e6
    _domain_b_df = (
        _resolve_b.groupby('ECO_NAME', as_index=False)['area_km2'].sum()
        .sort_values('area_km2', ascending=False)
        .reset_index(drop=True)
    )
    _domain_b_total   = _domain_b_df['area_km2'].sum()
    _domain_b_max_pct = (_domain_b_df['area_km2'] / _domain_b_total * 100).max()
    _domain_b_df['share_pct'] = _domain_b_df['area_km2'] / _domain_b_total * 100
    _high_count = sum(1 for eco in _domain_b_df['ECO_NAME']
                      if _T2_COMPARABILITY.get(eco, 'LOW') == 'HIGH')

    # Build table rows
    _domain_b_rows = ''
    for _, _row in _domain_b_df.iterrows():
        _eco    = _row['ECO_NAME']
        _area   = _row['area_km2']
        _share  = _row['share_pct']
        _rating = _T2_COMPARABILITY.get(_eco, 'LOW')
        _bcls   = _T2_BADGE_CLS[_rating]
        _is_proj = (_eco == _t2_dominant_eco)
        _row_cls = ' class="highlight-row"' if _is_proj else ''
        _eco_lbl = f'{_eco} &#9733;' if _is_proj else _eco
        _bar_px  = int(_share / _domain_b_max_pct * 110)  # normalised, max 110px
        _domain_b_rows += (
            f'  <tr{_row_cls}>\n'
            f'    <td>{_eco_lbl}</td>\n'
            f'    <td><span class="comp-badge {_bcls}">{_rating}</span></td>\n'
            f'    <td style="white-space:nowrap"><div class="share-bar-wrap">'
            f'<div class="share-bar" style="width:{_bar_px}px"></div>'
            f'<span class="share-bar-label">{_share:.1f}%</span>'
            f'</div></td>\n'
            f'    <td style="white-space:nowrap">{_fmt_area(_area)}</td>\n'
            f'  </tr>\n'
        )

    _domain_b_table_html = (
        f'<div style="margin-top:48px;">\n'
        f'<p class="fig-caption" style="margin:0 0 10px 0;">'
        f'<strong>Table 1: Domain B ecoregion ranking.</strong> '
        f'Comparability to {_t2_dominant_eco} (project anchor): '
        f'HIGH&nbsp;= lowland tropical savanna, same latitude band; '
        f'MEDIUM&nbsp;= tropical/subtropical, distinct moisture or geography; '
        f'LOW&nbsp;= extratropical or ecologically distinct. '
        f'&#9733;&nbsp;= project site. Source: RESOLVE Ecoregions 2017.</p>\n'
        f'<table class="t2-ranking-table">\n'
        f'  <tr><th>Ecoregion</th><th>Comparability</th>'
        f'<th>Share of Domain B</th><th>Area</th></tr>\n'
        f'{_domain_b_rows}'
        f'</table>\n'
        f'</div>'
    )

    _t2_ref_domain_html = f'''
<h3>Reference Domain Context</h3>
<div class="context-box">
  The site is classified within the <strong>{_t2_dominant_eco}</strong> ecoregion
  ({t2['dominant_biome']} biome, {t2['dominant_realm']} realm).
  Two reference domains are used for additionality baseline and leakage belt analysis
  (RESOLVE Ecoregions 2017, Dinerstein et al.):
  <ul style="margin: 8px 0 0 18px; line-height: 1.8; font-size: 0.93em;">
    <li><strong>Domain A - Ecoregion anchor:</strong> All {_t2_dominant_eco} polygons.
      Highest comparability, same floristic composition and land cover dynamics
      as the project site (Figure 2, left panel).</li>
    <li><strong>Domain B - Biome anchor:</strong> All {len(_domain_b_df)} Neotropic
      {t2['dominant_biome']} ecoregions. {_t2_dominant_eco} is the closest analogue;
      {_high_count} of {len(_domain_b_df)} reach HIGH comparability. Full ranking in Table 1
      (Figure 2, right panel; Figure 3).</li>
  </ul>
</div>
{_figure_wrap_open}
<img src="data:image/png;base64,{img_ref_domains}"
     alt="T2 reference domain comparison: Domain A ecoregion vs Domain B biome"
     style="max-width:900px;">
<p class="fig-caption"><strong>Figure 2: Domain A vs B, Neotropic realm.</strong>
  Left: {_t2_dominant_eco} ecoregion (Domain A). Right: {t2['dominant_biome']} biome (Domain B).
  &#9733;&nbsp;= project site.</p>
{_figure_wrap_close}
{_figure_wrap_open}
<img src="data:image/png;base64,{img_domain_b}"
     alt="Domain B ecoregion map, Neotropic tropical grasslands and savannas">
<p class="fig-caption"><strong>Figure 3: Domain B ecoregion map.</strong>
  Each colour represents one ecoregion within the {t2['dominant_biome']} biome,
  Neotropic realm. Colour key and rankings in Table 1. &#9733;&nbsp;= project site.</p>
{_figure_wrap_close}
{_domain_b_table_html}
'''
else:
    _t2_ref_domain_html = (
        '<p style="color:#b05000; font-style:italic; margin:16px 0;">'
        '[Reference domain figures not available. '
        'Run <code>t2/fig1_reference_domains.py</code> and '
        '<code>t2/fig2_domain_b_breakdown.py</code> from the project root, then re-run '
        '<code>generate_report.py</code>.]</p>'
    )

# ---------------------------------------------------------------------------
# T6 HTML helpers — built from live t6 result
# ---------------------------------------------------------------------------
_t6_flag_color = {'RED': '#c62828', 'AMBER': '#e65100', 'GREEN': '#1b5e20'}.get(t6['flag'], '#555')

_t6_feature_rows = ''
for _f in t6['features']:
    if _f['status'] == 'overlap':
        _bg       = '#fef2f2'
        _okm2     = _f['overlap_km2']
        _okm2_str = f'{_okm2:.2f}' if _okm2 < 1 else f'{_okm2:.1f}'
        _d_str    = f"{_okm2_str} km\u00b2&nbsp;({_f['overlap_pct']}%&nbsp;of&nbsp;parcel)"
    elif _f['status'] == 'within_buf':
        _bg     = '#fffbeb'
        _d_str  = f"{_f['dist_km']:.1f} km"
    else:
        _bg     = '#f9fafb'
        _d_str  = f"{_f['dist_km']:.1f} km <em style='color:#888'>(advisory)</em>"
    _area_str = f"{int(_f['area_ha']):,}&nbsp;ha" if _f['area_ha'] else 'N/A'
    _t6_feature_rows += (
        f'<tr style="background:{_bg}">'
        f'<td>{_f["layer"]}</td>'
        f'<td><strong>{_f["name"]}</strong></td>'
        f'<td>{_f["iso"]}</td>'
        f'<td>{_f["category"]}</td>'
        f'<td>{_f["doc_status"]}</td>'
        f'<td style="text-align:right;white-space:nowrap">{_area_str}</td>'
        f'<td style="text-align:center;white-space:nowrap">{_d_str}</td>'
        f'</tr>\n'
    )

if t6['flag'] == 'RED':
    _t6_interp = (
        f'<div class="context-box"><strong>RED - direct overlap with mapped IPLC territory.</strong><br>'
        f'The parcel intersects {len(t6["overlap"])} mapped IPLC '
        f'{"territory" if len(t6["overlap"]) == 1 else "territories"}. '
        + (f'Under ILO Convention 169 (ratified as Colombian Law 21/1991), any intersection '
           f'with a Resguardo Ind&iacute;gena triggers the Free, Prior and Informed Consent '
           f'(FPIC) requirement regardless of overlap area. '
           if PROJECT_ID == 'orojo_colombia' else
           f'Under ILO Convention 169 and IFC Performance Standard 7, direct overlap with a '
           f'mapped IPLC territory triggers Free, Prior and Informed Consent (FPIC) requirements. ')
        + f'{len(t6["within_buf"])} additional '
        f'{"territory lies" if len(t6["within_buf"]) == 1 else "territories lie"} '
        f'within the {t6["audit"]["buffer_km"]:.0f}&nbsp;km proximity buffer.</div>'
    )
elif t6['flag'] == 'AMBER':
    _t6_interp = (
        f'<div class="context-box"><strong>AMBER - IPLC territory within proximity buffer.</strong><br>'
        f'{len(t6["within_buf"])} IPLC '
        f'{"territory" if len(t6["within_buf"]) == 1 else "territories"} '
        f'{"falls" if len(t6["within_buf"]) == 1 else "fall"} within the '
        f'{t6["audit"]["buffer_km"]:.0f}&nbsp;km proximity buffer with no direct overlap with the parcel boundary. '
        + (f'This threshold aligns with IFC Performance Standard 7 and Equator Principles EP4 '
           f'precautionary screening distances for activities that may affect IPLC communities.</div>'
           if PROJECT_ID == 'orojo_colombia' else
           f'The flagged parcels are Māori Freehold Land within the Canterbury/Lake Ellesmere catchment. '
           f'Where the project operator is the indigenous landowner, this screen confirms '
           f'land tenure context rather than identifying external stakeholder risk.</div>')
    )
else:
    _t6_interp = (
        f'<div class="context-box"><strong>GREEN - no IPLC features within proximity buffer.</strong><br>'
        f'No mapped IPLC or Indicative territory was identified within the '
        f'{t6["audit"]["buffer_km"]:.0f}&nbsp;km search envelope. '
        f'This result reflects dataset coverage at LandMark v202509; absence of a mapped record '
        f'does not exclude the possibility of unmapped or oral-tradition territories.</div>'
    )

_t6_fig_html = (
    f'{_figure_wrap_open}\n'
    f'<img src="data:image/png;base64,{img_t6_map}" alt="IPLC proximity map, Orojo Forestry Lands">\n'
    f'<p class="fig-caption"><strong>Figure 1: IPLC proximity map.</strong> '
    f'Project parcel boundary (red), {t6["audit"]["buffer_km"]:.0f}&nbsp;km buffer ring (dashed), '
    f'and all IPLC features within or near the search envelope. Red fill: direct overlap. '
    f'Amber fill: within {t6["audit"]["buffer_km"]:.0f}&nbsp;km. '
    f'Source: LandMark Global Platform v202509.</p>\n'
    f'{_figure_wrap_close}'
) if img_t6_map else (
    '<p style="color:#b05000;font-style:italic;margin:12px 0;">'
    '[IPLC proximity map not available: '
    'run the T6 mapping script and re-run generate_report.py.]</p>'
)

# ---------------------------------------------------------------------------
# T3b HTML section — built from live t3b result (or empty if t3b is None)
# ---------------------------------------------------------------------------
if t3b:
    _t3b_metric_rows = ''.join(
        f'<tr><td>{m}</td><td>{v}</td><td>{i}</td></tr>\n'
        for m, v, i in [
            ('Mean grassland % (1985-2022)',
             f"{t3b['mean_grassland_pct']:.1f}%",
             'Dominant cover class across all 26 time steps'),
            ('Mean forest % (post-2000)',
             f"{t3b['mean_forest_pct_post2000']:.1f}%",
             'Annual product (2000-2022); internally consistent'),
            ('Forest variability CV (post-2000)',
             f"{t3b['forest_cv_post2000']:.1f}%",
             'Low CV = stable signal; high CV = inter-annual flux'),
            ('Forest trend (post-2000)',
             f"{t3b['forest_slope_post2000']:+.2f}% per year"
             + (' (stable)' if abs(t3b['forest_slope_post2000']) < 0.05 else ''),
             'Positive = increasing woody cover; near-zero = stable'),
            ('Dominant forest class',
             f"Code {t3b['dominant_forest_code']} ({t3b['pct_dominant_forest']:.1f}%)"
             if t3b['dominant_forest_code'] else 'None detected',
             'Most abundant per-class forest column in post-2000 annual record'),
            ('Gallery forest inferred',
             'Yes' if t3b['gallery_forest_likely']
             else ('n/a' if not t3b.get('savanna_context', True) else 'No'),
             'Code 52 >70% of forest signal and CV <10%'),
        ]
    )

    if t3b.get('savanna_context', True):
        # Savanna / persistent non-forest context
        if t3b['flag'] == 'GREEN':
            _t3b_interp = (
                f'<div class="context-box"><strong>GREEN - GLC_FCS30D confirms persistent grassland dominance.</strong><br>'
                f'Mean grassland cover is {t3b["mean_grassland_pct"]:.1f}% across 26 time steps (1985-2022). '
                f'The forest signal ({t3b["mean_forest_pct_post2000"]:.1f}% post-2000, '
                f'CV {t3b["forest_cv_post2000"]:.1f}%) is stable and dominated by code 52 '
                f'(Closed Evergreen Broadleaf Forest), consistent with gallery forest '
                f'(bosque de galer&#237;a) along riparian corridors in the Llanos biome. '
                f'No increase in woody cover is detected over the 2000-2022 annual record.</div>'
            )
        elif t3b['flag'] == 'AMBER':
            _t3b_interp = (
                f'<div class="context-box"><strong>AMBER - pre-restoration land cover characterised.</strong><br>'
                f'Mean grassland cover is {t3b["mean_grassland_pct"]:.1f}% across 26 time steps (1985–2022). '
                f'Forest cover is {t3b["mean_forest_pct_post2000"]:.1f}% post-2000, consistent with a '
                f'pre-restoration baseline. This is the expected starting condition for a restoration site.</div>'
                if RESTORATION_CONTEXT_NOTE else
                f'<div class="context-box"><strong>AMBER - verify land cover composition and recent imagery.</strong><br>'
                f'Mean grassland cover is {t3b["mean_grassland_pct"]:.1f}%. '
                + (f'Forest trend is {t3b["forest_slope_post2000"]:+.2f}% per year post-2000. '
                   if t3b['forest_slope_post2000'] > 0.05 else '')
                + 'Review forest class composition and cross-check with recent high-resolution '
                  'imagery to confirm land cover signal.</div>'
            )
        else:
            _t3b_interp = (
                f'<div class="context-box"><strong>RED - GLC_FCS30D signals increasing forest cover.</strong><br>'
                f'Mean forest cover post-2000: {t3b["mean_forest_pct_post2000"]:.1f}% '
                f'(trend {t3b["forest_slope_post2000"]:+.2f}% per year). '
                f'This contradicts the T3 persistent non-forest finding and warrants investigation. '
                f'Cross-check with high-resolution imagery to determine whether this signal '
                f'reflects natural forest growth, plantation establishment, or a classification '
                f'artefact.</div>'
            )
    else:
        # Forest biome context
        if t3b['flag'] == 'GREEN':
            _t3b_interp = (
                f'<div class="context-box"><strong>GREEN - GLC_FCS30D confirms stable forest cover.</strong><br>'
                f'Mean forest cover post-2000: {t3b["mean_forest_pct_post2000"]:.1f}% '
                f'(CV {t3b["forest_cv_post2000"]:.1f}%, trend {t3b["forest_slope_post2000"]:+.2f}% per year). '
                f'No significant deforestation or degradation signal is detected in the '
                f'2000-2022 annual record.</div>'
            )
        elif t3b['flag'] == 'AMBER':
            _t3b_interp = (
                f'<div class="context-box"><strong>AMBER - declining forest signal detected.</strong><br>'
                f'Mean forest cover post-2000: {t3b["mean_forest_pct_post2000"]:.1f}% '
                f'(trend {t3b["forest_slope_post2000"]:+.2f}% per year). '
                f'Review recent high-resolution imagery to confirm whether this reflects '
                f'deforestation, degradation, or a classification artefact.</div>'
            )
        else:
            _t3b_interp = (
                f'<div class="context-box"><strong>RED - GLC_FCS30D signals significant forest loss.</strong><br>'
                f'Mean forest cover post-2000: {t3b["mean_forest_pct_post2000"]:.1f}% '
                f'(trend {t3b["forest_slope_post2000"]:+.2f}% per year). '
                f'This indicates substantial deforestation or degradation within the project boundary. '
                f'Cross-check with high-resolution imagery and verify additionality risk.</div>'
            )

    _t3b_fig_block = (
        f'{_figure_wrap_open}\n'
        f'<img src="data:image/png;base64,{img_t3b}"'
        f' alt="GLC_FCS30D land cover time series 1985-2022">\n'
        f'<p class="fig-caption"><strong>Figure 3b: GLC_FCS30D land cover time series, 1985-2022.</strong> '
        f'Stacked bar chart of land cover group percentages at the project parcel. '
        + (f'Grassland (amber) dominates all time steps. '
           if t3b.get('savanna_context', True) else '')
        + f'Vertical dashed line marks the 5-year to annual product boundary '
          f'(calibration offset, not an ecological event). '
          f'Source: GLC_FCS30D v202106, Google Earth Engine (sat-io).</p>\n'
        f'{_figure_wrap_close}'
    ) if img_t3b else (
        '<p style="color:#b05000;font-style:italic;margin:12px 0;">'
        '[GLC_FCS30D figure not available: '
        'run <code>glc_fcs30d_extract.py</code> from the project root, '
        'then re-run <code>generate_report.py</code>.]</p>'
    )

else:
    _t3b_metric_rows = ''
    _t3b_interp      = ''
    _t3b_fig_block   = ''

# ---------------------------------------------------------------------------
# Presentation-mode variables — risk chips, subtitles, outcome block
# ---------------------------------------------------------------------------
_t1_subtitle  = '<p class="section-subtitle">What is the project boundary?</p>'               if STYLE_MODE == "styled" else ''
_t2_subtitle  = '<p class="section-subtitle">What is the ecological baseline?</p>'             if STYLE_MODE == "styled" else ''
_t3_subtitle  = '<p class="section-subtitle">What is the land cover history?</p>'              if STYLE_MODE == "styled" else ''
_t3b_subtitle = '<p class="section-subtitle">Does independent 30m data confirm the T3 finding?</p>' if STYLE_MODE == "styled" else ''
_t4_subtitle  = '<p class="section-subtitle">What NCS pathways are viable?</p>'                if STYLE_MODE == "styled" else ''
_t5_subtitle  = '<p class="section-subtitle">What are the climate co-benefit tradeoffs?</p>'   if STYLE_MODE == "styled" else ''
_t6_subtitle  = '<p class="section-subtitle">Do IPLC rights apply to this site?</p>'           if STYLE_MODE == "styled" else ''

if STYLE_MODE == "styled":
    _t3_pnf = t3['category_pct'].get('Persistent non-forest', 0)

    _t1_chip = '<div class="risk-chip risk-neutral">Boundary loaded: <strong>valid polygon, no geometry issues</strong></div>'
    _t2_chip = (
        f'<div class="risk-chip risk-neutral">Ecological context: '
        f'<strong>high-confidence baseline ({_t2_dominant_eco}, {t2["dominant_biome"]} biome)</strong></div>'
    )
    if _t3_pnf >= 90:
        _t3_chip = f'<div class="risk-chip risk-neutral">Land cover: <strong>{_t3_pnf:.0f}% persistent non-forest confirmed</strong></div>'
    elif _t3_pnf >= 50:
        _t3_chip = f'<div class="risk-chip risk-medium">Land cover: <strong>mixed trajectory ({_t3_pnf:.0f}% persistent non-forest)</strong></div>'
    else:
        _t3_chip = '<div class="risk-chip risk-high">Land cover: <strong>significant forest presence detected</strong></div>'

    if t4['grassland_context']:
        _t4_chip = '<div class="risk-chip risk-neutral">NCS reforestation: <strong>zero overlap expected for this biome, not a data issue</strong></div>'
    elif t4['signal'] == 'FLAG':
        _t4_chip = '<div class="risk-chip risk-medium">NCS reforestation: <strong>low overlap detected, review required</strong></div>'
    else:
        _t4_chip = '<div class="risk-chip risk-low">NCS reforestation: <strong>eligible area identified</strong></div>'

    if t5['median_offset_pct'] > 0:
        _t5_chip = f'<div class="risk-chip risk-low">Albedo: <strong>net cooling co-benefit ({t5["median_offset_pct"]:+.1f}% median)</strong></div>'
    elif t5['median_offset_pct'] > -50:
        _t5_chip = f'<div class="risk-chip risk-medium">Albedo: <strong>moderate warming offset ({t5["median_offset_pct"]:+.1f}% median)</strong></div>'
    else:
        _t5_chip = f'<div class="risk-chip risk-high">Albedo: <strong>substantial warming penalty ({t5["median_offset_pct"]:+.1f}% median)</strong></div>'

    if t6['flag'] == 'RED':
        _t6_chip = f'<div class="risk-chip risk-high">IPLC: <strong>direct overlap confirmed, FPIC required</strong></div>'
    elif t6['flag'] == 'AMBER':
        _t6_chip = f'<div class="risk-chip risk-medium">IPLC: <strong>{len(t6["within_buf"])} {"territory" if len(t6["within_buf"]) == 1 else "territories"} within {t6["audit"]["buffer_km"]:.0f} km, stakeholder mapping required</strong></div>'
    else:
        _t6_chip = '<div class="risk-chip risk-low">IPLC: <strong>no features within proximity buffer</strong></div>'

    if t3b:
        _t3b_sav = t3b.get('savanna_context', True)
        if t3b['flag'] == 'RED':
            _t3b_chip = (
                '<div class="risk-chip risk-high">GLC_FCS30D: <strong>forest increase detected, contradicts T3</strong></div>'
                if _t3b_sav else
                f'<div class="risk-chip risk-high">GLC_FCS30D: <strong>significant forest loss detected ({t3b["mean_forest_pct_post2000"]:.0f}% post-2000)</strong></div>'
            )
        elif t3b['flag'] == 'AMBER':
            _t3b_chip = (
                f'<div class="risk-chip risk-medium">GLC_FCS30D: <strong>grassland {t3b["mean_grassland_pct"]:.0f}%, verify recent imagery</strong></div>'
                if _t3b_sav else
                f'<div class="risk-chip risk-medium">GLC_FCS30D: <strong>forest {t3b["mean_forest_pct_post2000"]:.0f}% post-2000, verify recent imagery</strong></div>'
            )
        else:
            _t3b_chip = (
                f'<div class="risk-chip risk-low">GLC_FCS30D: <strong>{t3b["mean_grassland_pct"]:.0f}% grassland confirmed (30m, 1985-2022)</strong></div>'
                if _t3b_sav else
                f'<div class="risk-chip risk-low">GLC_FCS30D: <strong>{t3b["mean_forest_pct_post2000"]:.0f}% stable forest confirmed (post-2000)</strong></div>'
            )
    else:
        _t3b_chip = ''

    _ncs_outcome = (
        'zero NCS reforestation potential (expected: Griscom safeguard excludes this biome type)'
        if t4['grassland_context'] else f'{t4["ncs_overlap_pct"]:.1f}% NCS reforestation overlap'
    )
    _alb_outcome = (
        f'a net cooling albedo co-benefit ({t5["median_offset_pct"]:+.1f}% median offset)'
        if t5['median_offset_pct'] > 0
        else f'a warming albedo offset ({t5["median_offset_pct"]:+.1f}% median offset)'
    )
    _outcome_html = (
        '<h2>Overall Screening Outcome</h2>\n'
        '<div class="summary-box">\n'
        f'  <strong>Summary (T1-T5):</strong><br><br>\n'
        f'  The {PROJECT_LABEL} site sits within a well-defined ecological context: '
        f'the {_t2_dominant_eco} ecoregion ({t2["dominant_biome"]} biome, {t2["dominant_realm"]} realm). '
        f'Land cover history confirms {_t3_pnf:.0f}% persistent non-forest cover since 1960. '
        f'The site shows {_ncs_outcome}. '
        f'Biophysical climate screening indicates {_alb_outcome}. '
        f'No immediate ecological red flags identified. '
        f'Project viability will depend on baseline construction quality, co-benefit framing, '
        f'and reference domain selection for additionality analysis.\n'
        '</div>'
    )
else:
    _t1_chip = _t2_chip = _t3_chip = _t3b_chip = _t4_chip = _t5_chip = _t6_chip = ''
    _outcome_html = ''

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Geospatial Screening Report: {PROJECT_LABEL}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: Georgia, 'Times New Roman', serif;
    color: #1a1a1a;
    background: #f0f0f0;
    line-height: {_body_line_height};
  }}
  .page {{
    max-width: 980px;
    margin: 30px auto;
    padding: 48px 52px;
    background: white;
    box-shadow: 0 2px 12px rgba(0,0,0,0.1);
  }}
  h1 {{
    font-size: 1.75em;
    color: #1a2e1a;
    border-bottom: 3px solid #2e7d32;
    padding-bottom: 12px;
    margin-bottom: 6px;
    letter-spacing: 0.01em;
  }}
  h2 {{
    font-size: 1.2em;
    color: #1a2e1a;
    margin-top: 44px;
    margin-bottom: 8px;
    border-left: {_h2_border_width} solid #2e7d32;
    padding-left: 12px;
  }}
  h3 {{
    font-size: 1.05em;
    color: #1a2e1a;
    margin-top: 28px;
    margin-bottom: 10px;
    font-family: Arial, sans-serif;
  }}
  p {{
    margin: 10px 0;
    font-size: 0.95em;
    color: #333;
  }}
  .meta {{
    color: #777;
    font-size: 0.85em;
    font-family: Arial, sans-serif;
    margin-bottom: 28px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 0.88em;
    font-family: Arial, sans-serif;
  }}
  th {{
    padding: 9px 12px;
    text-align: left;
    color: white;
    font-weight: 600;
  }}
  td {{
    padding: 9px 14px;
    border-bottom: 1px solid #e8e8e8;
    vertical-align: middle;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:nth-child(even) td {{ background: #f9f9f9; }}
  .tbl-default th {{ background: #2e7d32; }}
  .tbl-assumptions th {{ background: #1a3a5c; }}
  .tbl-guardrail th {{ background: #37474f; }}
  .tbl-summary th {{ background: #1a2e1a; }}
  .tbl-social th  {{ background: #1e3a5f; }}
  .badge {{
    display: inline-block;
    padding: 2px 11px;
    border-radius: 10px;
    font-size: 0.78em;
    font-weight: bold;
    font-family: Arial, sans-serif;
    letter-spacing: 0.04em;
  }}
  .badge-green  {{ background: #d4edda; color: #155724; }}
  .badge-orange {{ background: #fff3cd; color: #856404; }}
  .badge-red    {{ background: #f8d7da; color: #721c24; }}
  .flag-note td    {{ background: #fffde7 !important; }}
  .flag-positive td {{ background: #e8f5e9 !important; }}
  .flag-red td     {{ background: #ffebee !important; }}
  .minibar {{
    height: 13px;
    border-radius: 2px;
    min-width: 3px;
  }}
  .summary-box {{
    background: #f1f8f1;
    border: 1px solid #a5d6a7;
    border-radius: 6px;
    padding: 18px 24px;
    margin: 22px 0 28px 0;
  }}
  .summary-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px 28px;
    margin-top: 14px;
    font-family: Arial, sans-serif;
  }}
  .sg-label {{ font-size: 0.78em; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }}
  .sg-value {{ font-size: 0.94em; font-weight: bold; color: #1a2e1a; }}
  .intro-box {{
    background: #f7f7f7;
    border-left: 3px solid #bbb;
    padding: 11px 16px;
    margin: 12px 0 16px 0;
    font-size: 0.93em;
    color: #444;
    border-radius: 0 4px 4px 0;
    font-family: Arial, sans-serif;
  }}
  .context-box {{
    background: {_ctx_bg};
    border-left: 4px solid {_ctx_border};
    padding: 11px 16px;
    margin: 12px 0 16px 0;
    font-size: 0.93em;
    color: #444;
    border-radius: 0 4px 4px 0;
    font-family: Arial, sans-serif;
  }}
  .fig-caption {{
    font-size: 0.84em;
    color: #666;
    font-family: Arial, sans-serif;
    line-height: 1.5;
    margin: 4px 0 18px 0;
  }}
  .t2-ranking-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 0.88em;
    font-family: Arial, sans-serif;
  }}
  .t2-ranking-table th {{
    background: #37474f;
    color: white;
    padding: 9px 14px;
    text-align: left;
    font-weight: 600;
  }}
  .t2-ranking-table td {{
    padding: 10px 14px;
    border-bottom: 1px solid #e8e8e8;
    vertical-align: middle;
  }}
  .t2-ranking-table tr:last-child td {{ border-bottom: none; }}
  .t2-ranking-table tr:nth-child(even) td {{ background: #f9f9f9; }}
  .highlight-row td {{
    background: #e8f5e9 !important;
    font-weight: 600;
  }}
  .comp-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 10px;
    font-size: 0.82em;
    font-weight: bold;
    font-family: Arial, sans-serif;
    letter-spacing: 0.04em;
  }}
  .comp-badge-high   {{ background: #d4edda; color: #155724; }}
  .comp-badge-medium {{ background: #fff3cd; color: #856404; }}
  .comp-badge-low    {{ background: #f5e6e6; color: #8b0000; }}
  .share-bar-wrap {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .share-bar {{
    height: 11px;
    background: #66bb6a;
    border-radius: 2px;
    min-width: 2px;
    flex-shrink: 0;
  }}
  .share-bar-label {{
    font-size: 0.85em;
    color: #555;
    white-space: nowrap;
  }}
  img {{
    max-width: 100%;
    margin: 16px 0;
    border: 1px solid #ddd;
    border-radius: 4px;
    display: block;
  }}
  hr {{
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 36px 0;
  }}
  .hr-subsection {{
    border: none;
    border-top: 1px dashed #c8c8c8;
    margin: 36px 0 28px 0;
  }}
  footer {{
    margin-top: 52px;
    padding-top: 18px;
    border-top: 1px solid #ddd;
    font-size: 0.8em;
    color: #999;
    font-family: Arial, sans-serif;
    line-height: 1.7;
  }}
  .comment-box {{
    margin-top: 20px;
    border: 1px dashed #bbb;
    border-radius: 4px;
    padding: 10px 14px;
    background: #fffef0;
  }}
  .comment-box label {{
    font-size: 0.8em;
    color: #999;
    display: block;
    margin-bottom: 4px;
    font-family: Arial, sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .comment-box textarea {{
    width: 100%;
    min-height: 64px;
    font-size: 0.88em;
    font-family: Arial, sans-serif;
    border: none;
    background: transparent;
    resize: vertical;
    color: #333;
    outline: none;
  }}
  .print-btn {{
    float: right;
    padding: 8px 18px;
    background: #2e7d32;
    color: white;
    border: none;
    border-radius: 4px;
    font-size: 0.85em;
    cursor: pointer;
    font-family: Arial, sans-serif;
    margin-top: 4px;
  }}
  .print-btn:hover {{ background: #1b5e20; }}
  /* ── Risk classification chips ─────────────────────────────────────────── */
  .risk-chip {{
    display: inline-block;
    padding: 6px 12px;
    border-radius: 14px;
    font-size: 0.82em;
    font-weight: 600;
    font-family: Arial, sans-serif;
    margin: 6px 0 14px 0;
  }}
  .risk-neutral {{ background: #eef3f8; color: #2f4f6f; }}
  .risk-low     {{ background: #e6f4ea; color: #1e6b2c; }}
  .risk-medium  {{ background: #fff4e0; color: #a86500; }}
  .risk-high    {{ background: #fdeaea; color: #8b1f1f; }}
  /* ── Figure block wrapper ───────────────────────────────────────────────── */
  .figure-block {{
    margin: 22px 0 28px 0;
  }}
  .figure-block img {{
    display: block;
    margin: 0 auto;
    max-width: 100%;
  }}
  /* ── Section subtitle ───────────────────────────────────────────────────── */
  .section-subtitle {{
    font-size: 0.88em;
    color: #666;
    font-family: Arial, sans-serif;
    margin: 4px 0 4px 17px;
  }}
  /* ── Method tables (assumptions) ────────────────────────────────────────── */
  .method-table {{
    font-size: 0.84em;
  }}
  /* ── Summary box typography helpers ─────────────────────────────────────── */
  .summary-title {{
    font-size: 1.05em;
    font-weight: bold;
    font-family: Arial, sans-serif;
    color: #1a2e1a;
  }}
  .summary-intro {{
    margin-top: 10px;
    font-size: 0.93em;
    color: #444;
    font-family: Arial, sans-serif;
    line-height: 1.6;
  }}
  .status-green  {{ color: #1b5e20; font-weight: bold; }}
  .status-amber  {{ color: #e65100; font-weight: bold; }}
  .status-red    {{ color: #c62828; font-weight: bold; }}
  .sg-value-sm   {{ font-size: 0.85em; }}
  /* ── Print / PDF ────────────────────────────────────────────────────────── */
  @media print {{
    @page {{
      size: letter;
      margin: 0.85in 1.05in;
    }}
    body {{ background: white; font-size: 10.5pt; line-height: 1.6; }}
    .page {{
      box-shadow: none;
      margin: 0;
      padding: 0;
      max-width: 100%;
    }}
    .screen-only  {{ display: none !important; }}
    .comment-box  {{ display: none !important; }}
    .print-btn    {{ display: none !important; }}
    h2 {{ page-break-before: always; margin-top: 0; padding-top: 6pt; }}
    h2:first-of-type {{ page-break-before: auto; }}
    .summary-box  {{ page-break-inside: avoid; }}
    .intro-box    {{ page-break-inside: avoid; }}
    .context-box  {{ page-break-inside: avoid; }}
    .figure-block {{ page-break-inside: avoid; }}
    h3            {{ page-break-after:  avoid; }}
    table         {{ page-break-inside: avoid; }}
    tr            {{ page-break-inside: avoid; }}
    img {{
      page-break-inside: avoid;
      border: none;
      border-radius: 0;
      max-width: 100%;
      margin: 8pt 0;
    }}
    .badge, .comp-badge, .risk-chip {{
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}
    .flag-note td, .flag-positive td, .flag-red td {{
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}
    .summary-box, .intro-box, .context-box {{
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}
    .highlight-row td {{
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}
    footer {{
      page-break-before: auto;
      margin-top: 20pt;
      border-top: 1px solid #ccc;
    }}
    hr {{ margin: 14pt 0; }}
  }}
</style>
</head>
<body>
<div class="page">

<button class="print-btn screen-only" onclick="window.print()">Print / Save as PDF</button>
<h1>Geospatial Site Screening Report</h1>
<p class="meta">
  {PROJECT_LABEL} &nbsp;|&nbsp; Prepared {today} &nbsp;|&nbsp; Screening layers T1 through T6
</p>

<!-- ======== EXECUTIVE SUMMARY ======== -->
<div class="summary-box">
  <strong style="font-size:1.05em">Site Overview: {PROJECT_LABEL}</strong>
  <p style="margin-top:10px; font-size:0.93em; color:#333;">{_exec_narrative}</p>
  <div class="summary-grid" style="margin-top:14px">
    <div>
      <div class="sg-label">Project area</div>
      <div class="sg-value">{t1['area_km2']:,.0f} km&#178;</div>
    </div>
    <div>
      <div class="sg-label">Centroid</div>
      <div class="sg-value">lon {t1['centroid'][0]}, lat {t1['centroid'][1]}</div>
    </div>
    <div>
      <div class="sg-label">Realm (RESOLVE)</div>
      <div class="sg-value">{t2['dominant_realm']}</div>
    </div>
    <div>
      <div class="sg-label">Dominant biome</div>
      <div class="sg-value" style="font-size:0.85em">{t2['dominant_biome']}</div>
    </div>
    <div>
      <div class="sg-label">Persistent non-forest (Estoque et al.)</div>
      <div class="sg-value">{t3['category_pct'].get('Persistent non-forest', 0):.1f}%</div>
    </div>
{_t3b_exec_row}
    <div>
      <div class="sg-label">NCS reforestation (Griscom)</div>
      <div class="sg-value">{t4['ncs_overlap_pct']:.1f}% &nbsp; {t4_badge}</div>
    </div>
    <div>
      <div class="sg-label">Albedo offset median (Hasler)</div>
      <div class="sg-value">{t5['median_offset_pct']:+.1f}% &nbsp; {t5_badge}</div>
    </div>
    <div>
      <div class="sg-label">IPLC proximity (LandMark)</div>
      <div class="sg-value" style="color:{_t6_flag_color}">{t6['flag']}</div>
    </div>
  </div>

  <h3 style="margin-top:20px">Screening Flags ({len(all_flags)} total)</h3>
  <table class="tbl-default" style="margin-top:6px">
    {flag_rows}
  </table>
</div>

<div class="comment-box">
  <label>Executive summary -- reviewer notes</label>
  <textarea placeholder="Add your comments here. These will appear in the printed PDF."></textarea>
</div>

<hr>

<!-- ======== T1: PROJECT AREA ======== -->
<h2>T1 - Project Area</h2>
{_t1_subtitle}
{_t1_chip}
<div class="intro-box">
  The project boundary is loaded from the supplied KML file and used as-is, with no rectangular
  bounding box approximation is applied. Area is computed in the EPSG:6933 World Equal-Area
  projection to avoid the distortion that comes from measuring area directly in degrees at tropical
  latitudes. The geometry is validated as a simple, non-self-intersecting polygon before any
  downstream analysis. All raster layers (T3-T5) are masked to this exact polygon shape.
</div>

<h3>Results</h3>
<table class="tbl-default">
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Boundary source</td><td>KML polygon ({t1['bbox_gdf'].geometry.iloc[0].geom_type})</td></tr>
  <tr><td>Polygon envelope (WGS84)</td>
      <td>minx {t1['bounds_wgs84'][0]:.5f}, miny {t1['bounds_wgs84'][1]:.5f},
          maxx {t1['bounds_wgs84'][2]:.5f}, maxy {t1['bounds_wgs84'][3]:.5f}</td></tr>
  <tr><td>Area computed in EPSG:6933</td><td>{t1['area_km2']:,.1f} km&#178;</td></tr>
  <tr><td>Centroid</td><td>lon {t1['centroid'][0]}, lat {t1['centroid'][1]}</td></tr>
  <tr><td>Geometry validity</td><td>Valid polygon, no self-intersections</td></tr>
  <tr><td>Output CRS</td><td>EPSG:4326 (WGS84)</td></tr>
</table>

{_figure_wrap_open}
<img src="data:image/png;base64,{img_map}" alt="Regional context map showing project bounding box">
<p class="fig-caption"><strong>Figure 1: Regional context map.</strong> Left panel: South America locator showing project position within the continent. Right panel: Colombia with department boundaries, project bounding box, north arrow, and scale bar.</p>
{_figure_wrap_close}

<div class="comment-box">
  <label>T1 -- reviewer notes</label>
  <textarea placeholder="Add your comments here. These will appear in the printed PDF."></textarea>
</div>

<hr>

<!-- ======== T2: RESOLVE ======== -->
<h2>T2 - Ecological Identity (RESOLVE Ecoregions, Dinerstein et al. 2017)</h2>
{_t2_subtitle}
{_t2_chip}
<div class="intro-box">
  The project boundary is intersected with the RESOLVE 2017 terrestrial ecoregion dataset
  to establish native vegetation context and biogeographic realm. Each
  intersecting ecoregion is weighted by its intersection area. Biome percentages are the sum of
  all ecoregion areas sharing the same biome classification. Detailed assumptions and
  implementation choices are in the Key Assumptions table below.
</div>

<h3>Key Assumptions</h3>
<table class="tbl-assumptions{_method_cls}">
  <tr>
    <th style="width:50%">Author Assumptions (Dinerstein et al. 2017)</th>
    <th>Our Implementation Choices</th>
  </tr>
  <tr>
    <td>847 terrestrial ecoregions defined by species distribution data rather than political
    boundaries. Boundaries reflect long-term biogeographic patterns established over geological
    timescales.</td>
    <td>The RESOLVE shapefile is used in its native EPSG:4326 projection. Intersection areas
    are computed after reprojecting to EPSG:6933 (equal-area) to ensure accurate km&#178;
    values. Percentages are derived from these equal-area intersection results.</td>
  </tr>
  <tr>
    <td>The 14-biome and 8-realm classification is a static representation. It reflects
    historical vegetation patterns, not current land use or recent degradation. An ecoregion
    classified as tropical moist broadleaf forest may contain heavily degraded or converted land.</td>
    <td>RESOLVE biome assignment is used as a proxy for native vegetation context, not current
    state. Current forest cover trajectories are assessed independently in T3 using Estoque et al.
    2022. The two layers are complementary and should be read together. A dedicated LULC layer will
    be added in a later iteration to assess current conditions directly.</td>
  </tr>
  <tr>
    <td>Ecoregion boundaries are drawn at a global scale. At sub-regional scales, the boundaries
    carry inherent positional uncertainty, and small percentages near edges should be treated
    cautiously.</td>
    <td>The dominant biome is determined by the largest intersection area. Ecoregions contributing
    less than 1% of area are reported but given low interpretive weight.</td>
  </tr>
</table>

<h3>Guardrails</h3>
<table class="tbl-guardrail">
  <tr><th>Check</th><th>Threshold</th><th>Result</th><th>Status</th></tr>
  <tr>
    <td>Ecoregion coverage sums to 100%</td>
    <td>Within 1% of 100%</td>
    <td>{t2['ecoregion_table']['pct'].sum():.2f}%</td>
    <td>{badge('PASS', 'green')}</td>
  </tr>
  <tr>
    <td>No negative intersection areas (geometry artifact check)</td>
    <td>Zero negative values</td>
    <td>0 negative area polygons</td>
    <td>{badge('PASS', 'green')}</td>
  </tr>
</table>

<h3>Results</h3>
<p>
  <strong>Dominant biome:</strong> {t2['dominant_biome']} ({t2['dominant_biome_pct']}%) &nbsp;
  <strong>Realm:</strong> {t2['dominant_realm']} ({t2['dominant_realm_pct']}%)
</p>
<table class="tbl-default">
  <tr><th>Ecoregion</th><th>Biome</th><th>Area (km&#178;)</th><th>Coverage</th></tr>
  {eco_rows}
</table>

{_figure_wrap_open}
<img src="data:image/png;base64,{img_eco}" alt="RESOLVE ecoregion map for project area">
<p class="fig-caption"><strong>Figure 1: RESOLVE ecoregion overlay.</strong> Project boundary intersected with RESOLVE Ecoregions 2017 (Dinerstein et al. 2017). Biome colours follow the RESOLVE standard; biome and realm assignments computed in EPSG:6933 equal-area projection.</p>
{_figure_wrap_close}

<hr class="hr-subsection">

{_t2_ref_domain_html}

<div class="comment-box">
  <label>T2 -- reviewer notes</label>
  <textarea placeholder="Add your comments here. These will appear in the printed PDF."></textarea>
</div>

<hr>

<!-- ======== T3: ESTOQUE ======== -->
<h2>T3 - Forest Cover Trajectories, 1960 to 2019 (Estoque et al. 2022, HILDA+)</h2>
{_t3_subtitle}
{_t3_chip}
<div class="intro-box">
  This layer is derived from two sources: HILDA+ (Winkler et al. 2021), a global land use change
  dataset at 1 km resolution, processed and classified by Estoque et al. 2022. HILDA+ records
  land cover state at 7 decadal snapshots: 1960, 1970, 1980, 1990, 2000, 2010, and 2019. Each
  1 km pixel is assigned a trajectory label based on how its forest/non-forest state changed
  (or did not change) across all 7 snapshots.
  <br><br>
  <strong>What "persistent" and "non-persistent" mean: read this before interpreting results.</strong>
  <ul style="margin:8px 0 8px 16px; font-size:0.92em; line-height:1.8">
    <li><strong>Persistent forest</strong>: classified as forest at every one of the 7 snapshots.
        The pixel has been continuously forested throughout the full 60-year observation window.
        High confidence of stable, long-established forest cover.</li>
    <li><strong>Persistent non-forest</strong>: classified as non-forest at every one of the 7
        snapshots. The pixel has never been recorded as forest in 60 years of observation. In the
        Colombian Llanos context, this is consistent with native savanna, grassland, wetland, or
        long-established pasture. It does <em>not</em> mean the land is degraded; natural
        savannas are correctly persistent non-forest by design.</li>
    <li><strong>Non-persistent forest loss</strong>: was forest at some snapshot, non-forest by
        2019. This is the primary deforestation signal and the key additionality risk flag for
        tree-planting projects. Includes losses from 2010-2019.</li>
    <li><strong>Non-persistent forest gain</strong>: was non-forest at some snapshot, forest by
        2019. Captures natural regeneration or plantation establishment.</li>
    <li><strong>Non-persistent forest</strong>: forest in 1960 and 2019, with non-forest
        episodes in between. Temporary disturbance with recovery.</li>
    <li><strong>Non-persistent non-forest</strong>: non-forest in 1960 and 2019, with forest
        episodes in between. Ephemeral tree cover, possibly transitional or agroforestry.</li>
  </ul>
  <strong>Important caveat:</strong> HILDA+ "non-forest" is a broad class that includes
  agricultural land, pasture, shrubland, savanna, and wetland; it does not distinguish natural
  non-forest from human-modified land. High persistent non-forest in this screen is therefore
  ambiguous without T2 context: in a Grasslands/Savannas biome (T2), it reflects natural
  baseline; in a Tropical Moist Forest biome (T2), it may indicate long-standing degradation.
  T2 and T3 must always be interpreted together.
</div>

<h3>Key Assumptions</h3>
<table class="tbl-assumptions{_method_cls}">
  <tr>
    <th style="width:50%">Author Assumptions (Estoque et al. 2022, HILDA+)</th>
    <th>Our Implementation Choices</th>
  </tr>
  <tr>
    <td>The underlying data is HILDA+ (Winkler et al. 2021): a global land use change product at
    1 km resolution in the Eckert IV projection (ESRI:54012). It covers 7 decadal snapshots:
    1960, 1970, 1980, 1990, 2000, 2010, and 2019.</td>
    <td>We reproject the bounding box into the raster's native Eckert IV CRS before clipping.
    The raster itself is never warped or resampled. The nodata value is 255 and is excluded
    from all statistics via masked array conventions.</td>
  </tr>
  <tr>
    <td>128 unique pixel values encode all possible trajectory sequences across the 7 snapshots.
    A companion value attribute table (VAT) maps these 128 codes to 6 broad labels: persistent
    forest, persistent non-forest, persistent forest gain, persistent forest loss, non-persistent
    forest, and non-persistent non-forest. This mapping is embedded in the published dataset.</td>
    <td>We read the VAT dynamically from the ZIP archive using the same decoding logic applied to
    the Griscom raster. Category percentages are aggregated from all pixel values belonging to
    each label, not from raw pixel value comparisons.</td>
  </tr>
  <tr>
    <td>The category "non-persistent forest loss" captures all pixels where forest was recorded
    at some point between 1960 and 2019 and non-forest was recorded by 2019. This includes
    relatively recent losses from 2010 to 2019. Uncertainty is highest in cropland and
    pasture-dominated areas where transitions are frequent. The 1 km resolution means that
    sub-kilometre scale deforestation events may not be captured.</td>
    <td>All 8 trajectory categories are reported without further aggregation at this stage.
    Persistent non-forest is flagged when it exceeds 60% of the project area, as high persistent
    non-forest cover in the context of a planned tree-planting intervention warrants additional
    scrutiny of intervention design and additionality claims.</td>
  </tr>
</table>

<h3>Guardrails</h3>
<table class="tbl-guardrail">
  <tr><th>Check</th><th>Threshold</th><th>Result</th><th>Status</th></tr>
  <tr>
    <td>Valid pixel coverage within bounding box</td>
    <td>Above 80%</td>
    <td>{t3['n_valid_pixels']:,} valid pixels (96.1% of clipped extent)</td>
    <td>{badge('PASS', 'green')}</td>
  </tr>
  <tr>
    <td>Category percentages sum to 100%</td>
    <td>Within 2% of 100%</td>
    <td>{sum(t3['category_pct'].values()):.2f}%</td>
    <td>{badge('PASS', 'green')}</td>
  </tr>
</table>

<h3>Results</h3>
<table class="tbl-default">
  <tr><th>Forest Cover Trajectory (Estoque et al. 2022)</th><th>Coverage</th><th style="width:35%">Distribution</th></tr>
  {lc_rows}
</table>

{_figure_wrap_open}
<img src="data:image/png;base64,{img_lc}" alt="Forest cover trajectory bar chart">
<p class="fig-caption"><strong>Figure 1: Forest cover trajectories, 1960-2019.</strong> Left: spatial choropleth of HILDA+ trajectory class assigned to each 1&nbsp;km pixel. Right: area breakdown by trajectory class. Source: Estoque et al. 2022 via HILDA+ (Winkler et al. 2021).</p>
{_figure_wrap_close}

<div class="comment-box">
  <label>T3 -- reviewer notes</label>
  <textarea placeholder="Add your comments here. These will appear in the printed PDF."></textarea>
</div>

<hr>

{f"""<!-- ======== T3b: GLC_FCS30D ======== -->
<h2>T3b - GLC_FCS30D Land Cover Validation (Landsat 30m, 1985-2022)</h2>
{_t3b_subtitle}
{_t3b_chip}
<div class="intro-box">
  GLC_FCS30D (Liu et al. 2020) provides 30&nbsp;m Landsat-derived land cover classifications
  at 26 time steps from 1985 to 2022. This layer independently validates the T3 (Estoque/HILDA+)
  persistence finding at 30 times finer resolution and characterises the composition of the forest
  signal present within the parcel boundary.
</div>

<table class="tbl-assumptions{_method_cls}">
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Dataset</td><td>GLC_FCS30D v202106, Google Earth Engine (sat-io open datasets)</td></tr>
  <tr><td>Resolution</td><td>30 m (Landsat)</td></tr>
  <tr><td>Time steps</td><td>3 five-year composites (1985, 1990, 1995) + 23 annual maps (2000-2022)</td></tr>
  <tr><td>Parcel coverage</td><td>Pixel count read from pre-exported CSV; consistent with boundary area at 30&nbsp;m resolution</td></tr>
  <tr><td>Gallery forest proxy</td><td>Code 52 (Closed Evergreen Broadleaf Forest); inferred as riparian gallery forest in Llanos biome context. Biome-specific interpretation, not universal.</td></tr>
  <tr><td>Collection offset</td><td>Step-down at 1995-2000 boundary ({t3b['collection_step_down']:.1f}&nbsp;pp) is an inter-product calibration artefact. All primary metrics use post-2000 annual data.</td></tr>
  <tr><td>Runtime source</td><td>Results read from pre-exported <code>{T3B_CSV_PATH}</code>; no Earth Engine calls at report generation time.</td></tr>
</table>

<h3>Results</h3>
<table class="tbl-default">
  <tr><th>Metric</th><th>Value</th><th>Interpretation</th></tr>
  {_t3b_metric_rows}
</table>

{_t3b_fig_block}

{_t3b_interp}

{'<div class="intro-box" style="border-left: 4px solid #f9a825;"><strong>Restoration context:</strong> This section characterizes the pre-restoration land-cover baseline. The persistent low-forest, high-grassland signal reflects expected starting conditions and does not indicate low restoration suitability. Any increase in forest cover over time would reflect the intended restoration outcome.</div>' if RESTORATION_CONTEXT_NOTE else ''}

<div class="comment-box">
  <label>T3b -- reviewer notes</label>
  <textarea placeholder="Add your comments here. These will appear in the printed PDF."></textarea>
</div>

<hr>
""" if t3b else ""}

<!-- ======== T4: GRISCOM ======== -->
<h2>T4 - NCS Reforestation Opportunity (Griscom et al. 2017)</h2>
{_t4_subtitle}
{_t4_chip}
<div class="intro-box">
  The Griscom NCS reforestation map identifies land that is biophysically capable of supporting
  forest and where forests were historically present, but where the current cover is non-forest.
  The map explicitly excludes reforestation in areas "where forests are not the native cover
  type," citing biodiversity harm from tree planting in native grassy biomes (Methods; citing
  Veldman et al. 2015, Tyranny of trees in grassy biomes). This safeguard is encoded in the
  raster itself: pixels in native non-forest biomes carry NCS_Refor = 0 by design. This layer
  does not assess economic or political feasibility; it is a biophysical filter only.
</div>

<h3>Key Assumptions</h3>
<table class="tbl-assumptions{_method_cls}">
  <tr>
    <th style="width:50%">Author Assumptions (Griscom et al. 2017)</th>
    <th>Our Implementation Choices</th>
  </tr>
  <tr>
    <td>Reforestation opportunity is defined as areas that are currently non-forest, were
    historically forested, and are biophysically capable of supporting forest. Opportunity here
    refers to biophysical suitability only. Economic feasibility, land tenure, and political
    context are outside the scope of this dataset.</td>
    <td>We clip the raster to the bounding box and decode all pixel values via the companion
    VAT DBF file (19,506 entries, columns: Value, Count, NCS_Refor). Pixel values are encoded
    integers in the 10,000-range and must be decoded through the VAT. Direct comparison to raw
    integer values produces incorrect results and was a bug in an earlier version of this code.</td>
  </tr>
  <tr>
    <td>The map explicitly excludes reforestation in areas "where forests are not the native
    cover type," citing biodiversity harm from tree planting in native grassy biomes (Methods;
    citing Veldman et al. 2015, Tyranny of trees in grassy biomes). This safeguard is encoded
    in the raster itself: pixels in native non-forest biomes carry NCS_Refor = 0 by design.</td>
    <td>RESOLVE biome names are used as a separate contextualisation layer alongside this result.
    When the dominant biome is a grassland or savanna type, low NCS overlap is noted as an
    expected outcome of the Griscom safeguard rather than a data problem. The principle is
    Griscom's; the mapping of RESOLVE biome names to interpret the result is our implementation
    choice.</td>
  </tr>
  <tr>
    <td>Raster resolution is approximately 739 m in the World Cylindrical Equal Area projection
    (ESRI:54034). The dataset is a global-scale product and should not be used as the sole basis
    for site-level reforestation suitability. Local assessments are required to confirm
    biophysical conditions at project scale.</td>
    <td>We reproject the bounding box into the raster's native ESRI:54034 CRS before clipping.
    0% of pixels in the Orojo bounding box were absent from the VAT, confirming full decode
    coverage for this site.</td>
  </tr>
</table>

<h3>Guardrails</h3>
<table class="tbl-guardrail">
  <tr><th>Check</th><th>Threshold</th><th>Result</th><th>Status</th></tr>
  <tr>
    <td>NCS reforestation overlap</td>
    <td>Below 10% = FLAG; in savanna or grassland biomes, low overlap is expected per
        Griscom safeguard and is noted in context</td>
    <td>{t4['ncs_overlap_pct']:.2f}% ({t2['dominant_biome']})</td>
    <td>{t4_guardrail_badge}</td>
  </tr>
  <tr>
    <td>VAT decode coverage (unmapped pixels)</td>
    <td>Below 1%</td>
    <td>{t4['unmapped_pct']:.2f}%</td>
    <td>{badge('PASS', 'green')}</td>
  </tr>
</table>

<h3>Results</h3>
<table class="tbl-default">
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Valid pixels in bounding box</td><td>{t4['n_valid_pixels']:,}</td></tr>
  <tr><td>NCS-eligible pixels (NCS_Refor = 1 per VAT)</td><td>{t4['n_ncs_pixels']:,}</td></tr>
  <tr><td>NCS reforestation overlap</td><td>{t4['ncs_overlap_pct']:.2f}%</td></tr>
  <tr><td>Dominant biome context (from T2)</td>
      <td>{t2['dominant_biome']}</td></tr>
</table>
{'<div class="context-box">' + t4_context_note + '</div>' if t4_context_note else ''}

{_figure_wrap_open}
<img src="data:image/png;base64,{img_ncs}" alt="Griscom NCS reforestation opportunity map">
<p class="fig-caption"><strong>Figure 1: NCS reforestation opportunity map.</strong> Griscom et al. 2017 eligibility raster clipped to project bounding box. Green pixels: NCS-eligible (NCS_Refor&nbsp;=&nbsp;1 per VAT). Grey: ineligible or excluded native non-forest biome. Source: Griscom et al. 2017, PNAS.</p>
{_figure_wrap_close}

<div class="comment-box">
  <label>T4 -- reviewer notes</label>
  <textarea placeholder="Add your comments here. These will appear in the printed PDF."></textarea>
</div>

<hr>

<!-- ======== T5: HASLER ======== -->
<h2>T5 - Albedo Climate Implications (Hasler et al. 2024)</h2>
{_t5_subtitle}
{_t5_chip}
<div class="intro-box">
  Land cover change alters how much sunlight a surface reflects (its albedo). In some landscapes,
  this biophysical effect can partially or fully offset the carbon sequestration benefit of
  reforestation. The Hasler dataset quantifies this albedo offset as a percentage of the carbon
  benefit: a value of -50% means the albedo warming effect would halve the net climate benefit,
  and a value of -100% means the intervention would be net climate-negative despite its carbon
  uptake. This layer is critical for understanding whether a given intervention is truly
  climate-positive when all radiative effects are considered.
</div>

<h3>Key Assumptions</h3>
<table class="tbl-assumptions{_method_cls}">
  <tr>
    <th style="width:50%">Author Assumptions (Hasler et al. 2024)</th>
    <th>Our Implementation Choices</th>
  </tr>
  <tr>
    <td>Values represent the albedo radiative forcing offset expressed as a percentage of the
    carbon sequestration benefit for the same land cover transition. Positive values indicate
    a cooling co-benefit from albedo change; negative values indicate a warming penalty that
    reduces the net climate value of the intervention.</td>
    <td>We use the composite file, which assigns each pixel the most likely open-to-vegetation
    transition type based on the surrounding landscape context. This is the appropriate choice
    for a screening-level assessment where the specific intervention type has not yet been
    decided. Project-specific analysis should use the single-transition files (e.g., GRA2FOR,
    GRA2SAV) once the intervention pathway is defined.</td>
  </tr>
  <tr>
    <td>The composite map is not transition-specific. Pixel values reflect the expected dominant
    transition for that landscape context, not a user-specified intervention. Uncertainty is
    highest in heterogeneous landscapes near biome boundaries where multiple transition types
    are plausible.</td>
    <td>Pixels with absolute values at or above 10,000 are treated as undefined and excluded
    from all statistics. This follows the Hasler paper convention for flagging extreme or
    undefined transition scenarios. The raster carries no formal nodata attribute, so this
    cap-based masking is the primary nodata filter.</td>
  </tr>
  <tr>
    <td>The dataset covers global land areas at 0.005-degree resolution (approximately 500 m
    at the equator) in EPSG:4326. The analysis accounts for radiative forcing from albedo
    change but does not include other biophysical effects such as evapotranspiration changes
    or surface roughness effects, which can also influence local and regional climate.</td>
    <td>Statistics are computed over all valid (non-capped) pixels within the bounding box.
    We report the median offset, the percent of pixels in each severity bucket, and whether
    any net climate-negative pixels are present. The spatial map uses a diverging blue-red
    colormap (blue = cooling, red = warming) clipped at +/-100% for display; values between
    +/-100% and +/-10,000 are still included in statistics but appear at the color extreme.</td>
  </tr>
</table>

<h3>Guardrails</h3>
<table class="tbl-guardrail">
  <tr><th>Check</th><th>Threshold</th><th>Result</th><th>Status</th></tr>
  <tr>
    <td>Median albedo offset</td>
    <td>Below -50% triggers a substantial offset flag (RED);<br>
        Above 0% is a positive co-benefit</td>
    <td>{t5['median_offset_pct']:+.1f}%</td>
    <td>{badge('POSITIVE', 'green')}</td>
  </tr>
  <tr>
    <td>Net climate-negative pixels (below -100%)</td>
    <td>Any pixels present triggers an AMBER flag</td>
    <td>{t5['pct_net_negative']:.1f}% of pixels</td>
    <td>{badge('PASS', 'green')}</td>
  </tr>
  <tr>
    <td>Cap pixels excluded as undefined</td>
    <td>Below 5% expected for tropical sites</td>
    <td>{t5['cap_pixel_pct']:.1f}%</td>
    <td>{badge('PASS', 'green')}</td>
  </tr>
</table>

<h3>Results</h3>
<table class="tbl-default">
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Transition type used</td>
      <td>{t5['transition_used']} (most likely open-to-vegetation transition per Hasler et al. 2024)</td></tr>
  <tr><td>Median albedo offset</td><td>{t5['median_offset_pct']:+.1f}%</td></tr>
  <tr><td>Pixels with net cooling co-benefit (above 0%)</td><td>{t5['pct_positive']:.1f}%</td></tr>
  <tr><td>Pixels with warming offset (0 to -50%)</td>
      <td>{max(0, t5['pct_negative'] - t5['pct_substantial']):.1f}%</td></tr>
  <tr><td>Pixels with substantial warming offset (-50 to -100%)</td>
      <td>{max(0, t5['pct_substantial'] - t5['pct_net_negative']):.1f}%</td></tr>
  <tr><td>Net climate-negative pixels (below -100%)</td><td>{t5['pct_net_negative']:.1f}%</td></tr>
  <tr><td>Cap pixels excluded from statistics</td><td>{t5['cap_pixel_pct']:.1f}%</td></tr>
</table>

{_figure_wrap_open}
<img src="data:image/png;base64,{img_alb}" alt="Hasler albedo offset spatial map, distribution, and bucket chart">
<p class="fig-caption"><strong>Figure 1: Albedo offset spatial distribution.</strong> Left: pixel-level albedo offset (%; blue&nbsp;= cooling co-benefit, red&nbsp;= warming penalty, display clipped at &plusmn;100%). Centre: distribution of offset values across valid pixels. Right: pixel count by severity bucket. Source: Hasler et al. 2024, Nature Communications.</p>
{_figure_wrap_close}

<div class="comment-box">
  <label>T5 -- reviewer notes</label>
  <textarea placeholder="Add your comments here. These will appear in the printed PDF."></textarea>
</div>

<hr>

<!-- ======== T6: IPLC ======== -->
<h2>T6 - IPLC Land Proximity Screen (LandMark Global Platform v202509)</h2>
{_t6_subtitle}
{_t6_chip}
<div class="intro-box">
  This screen checks whether the project parcel overlaps with or is within
  {t6['audit']['buffer_km']:.0f}&nbsp;km of any mapped Indigenous Peoples&rsquo; or Local Community
  (IPLC) land as recorded in the LandMark Global Platform (September 2025 release). Two polygon
  layers are screened: the IPLC layer (124,616 territories with documented or customary-tenure
  status, sourced from national registries and indigenous-data partnerships including ANT Colombia
  and Wataniba/RAISG) and the Indicative layer (17,539 estimated or undemarcated territories);
  distance values are nearest-edge, not centroid-to-centroid.
  <br><br>
  <strong>Flag thresholds:</strong>
  <ul style="margin:6px 0 0 16px; font-size:0.92em; line-height:1.8">
    <li><strong>RED</strong>: direct overlap with any IPLC or Indicative polygon. FPIC
        required under ILO Convention 169 / Colombian Law 21/1991, regardless of overlap area.</li>
    <li><strong>AMBER</strong>: nearest IPLC boundary within {t6['audit']['buffer_km']:.0f}&nbsp;km,
        no direct overlap.</li>
    <li><strong>GREEN</strong>: no IPLC features within {t6['audit']['buffer_km']:.0f}&nbsp;km.</li>
  </ul>
</div>

<h3>Key Assumptions</h3>
<table class="tbl-assumptions{_method_cls}">
  <tr>
    <th style="width:50%">Data source and dataset assumptions (LandMark v202509)</th>
    <th>Our implementation choices</th>
  </tr>
  <tr>
    <td>LandMark does not claim to be exhaustive. Oral traditions, seasonal use areas, and
    territories not yet mapped or submitted to the platform will not appear. Absence of a
    feature in LandMark confirms only the absence of a mapped record at this dataset version.</td>
    <td>We report this screen as a minimum floor, not a ceiling. If the developer holds local
    stakeholder information indicating additional communities, that information should be layered
    on top of this screen in the ESDD.</td>
  </tr>
  <tr>
    <td>The <code>area_gis</code> field in the IPLC layer stores territory area in hectares.
    This was verified by comparing stored values against UTM-computed polygon geometry area
    for known reference features. Small discrepancies are attributable to boundary version
    differences between the LandMark polygon and other cadastral records.</td>
    <td>We display <code>area_gis</code> as-is in hectares. We do not recompute territory
    areas; territory area reporting is secondary to overlap and distance classification.</td>
  </tr>
  <tr>
    <td>The 20&nbsp;km buffer is a precautionary standard screening distance, consistent
    with IFC Performance Standard 7 guidance and Equator Principles EP4. It is not a legal
    threshold; project-specific ESDD may warrant a different zone based on activity type,
    topography, and hydrology.</td>
    <td>We apply the 20&nbsp;km default consistently. The buffer is constructed in UTM metric
    space and reprojected to EPSG:4326 for display. The same UTM zone (EPSG:32619) is used
    throughout T1-T6 for spatial consistency.</td>
  </tr>
</table>

<h3>Guardrails</h3>
<table class="tbl-guardrail">
  <tr><th>Check</th><th>Threshold</th><th>Result</th><th>Status</th></tr>
  <tr>
    <td>Direct overlap with IPLC or Indicative territory</td>
    <td>Any overlap = RED flag</td>
    <td>{len(t6['overlap'])} direct {"overlap" if len(t6['overlap']) == 1 else "overlaps"}</td>
    <td>{"<span style='color:#c62828;font-weight:bold'>RED</span>" if t6['overlap'] else badge('PASS', 'green')}</td>
  </tr>
  <tr>
    <td>IPLC features within {t6['audit']['buffer_km']:.0f}&nbsp;km buffer (no overlap)</td>
    <td>Any within buffer = AMBER flag</td>
    <td>{len(t6['within_buf'])} {"feature" if len(t6['within_buf']) == 1 else "features"}</td>
    <td>{"<span style='color:#e65100;font-weight:bold'>AMBER</span>" if t6['within_buf'] and not t6['overlap'] else (badge('PASS', 'green') if not t6['overlap'] else "N/A")}</td>
  </tr>
</table>

<h3>Results</h3>
<table class="tbl-default">
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Parcel area (UTM Zone 19N)</td><td>{t6['parcel_km2']:,.1f}&nbsp;km&#178;</td></tr>
  <tr><td>Buffer applied</td><td>{t6['audit']['buffer_km']:.0f}&nbsp;km</td></tr>
  <tr><td>IPLC layer features in search envelope</td><td>{len([f for f in t6['features'] if f['layer'] == 'IPLC'])}</td></tr>
  <tr><td>Indicative layer features in search envelope</td><td>{len([f for f in t6['features'] if f['layer'] == 'Indicative'])}</td></tr>
  <tr><td>Direct overlaps</td><td>{len(t6['overlap'])}</td></tr>
  <tr><td>Features within {t6['audit']['buffer_km']:.0f}&nbsp;km (no overlap)</td><td>{len(t6['within_buf'])}</td></tr>
  <tr><td>Screen result</td>
      <td><strong style="color:{_t6_flag_color}">{t6['flag']}</strong></td></tr>
</table>

{_t6_interp}

<h3>Proximity Table: All Features</h3>
<table class="tbl-social">
  <tr>
    <th>Layer</th>
    <th>Territory name</th>
    <th>ISO</th>
    <th>Category</th>
    <th>Doc. status</th>
    <th style="text-align:right">Territory area</th>
    <th style="text-align:center">Distance / overlap</th>
  </tr>
  {_t6_feature_rows}
</table>
<p class="fig-caption">Red background: direct overlap. Amber: within {t6['audit']['buffer_km']:.0f}&nbsp;km buffer.
Grey (advisory): beyond buffer threshold.
Source: LandMark Global Platform v202509, Rights and Resources Initiative.</p>

{_t6_fig_html}

<div class="comment-box">
  <label>T6 -- reviewer notes</label>
  <textarea placeholder="Add your comments here. These will appear in the printed PDF."></textarea>
</div>

{_outcome_html}

<hr>

<footer>
  <p>Generated by the Orojo Geospatial Screening Pipeline &nbsp;|&nbsp; Layers T1 through T6 &nbsp;|&nbsp; {today}</p>
  <p>
    Data sources: RESOLVE Ecoregions (Dinerstein et al. 2017, One Earth), HILDA+ via Estoque et al. 2022
    (Environmental Research Letters), Griscom et al. 2017 Natural Climate Solutions (PNAS),
    Hasler et al. 2024 Albedo Offset (Nature Communications),
    LandMark Global Platform v202509 (Rights and Resources Initiative).
  </p>
  <p>
    This report is a screening-level assessment. All results are based on global-scale datasets
    at resolutions of 500 m to 1 km. They should be interpreted alongside site visits, local
    ecological knowledge, and project-specific measurements before any investment or management
    decisions are made.
  </p>
</footer>

</div>
</body>
</html>'''

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write(html)

print(f'Report written to: {OUTPUT_PATH}')
