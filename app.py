import streamlit as st
import pandas as pd
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Dashboard Master: Cost vs Revenue", page_icon="💰", layout="wide")

# ==========================================
# ⚙️ CONFIGURACIÓN DE DATOS (YA ACTUALIZADA)
# ==========================================
SHEET_ID = "1WuBv1esTxZAfC07BPwWzjsz5TZqfUHa6MOzNIAEOMew"

# GIDs extraídos de tus enlaces:
GID_ANDROID = "368085162"    # Pestaña: Datos por AOS/dia
GID_IOS = "1225911759"       # Pestaña: Datos por IOS/dia
# ==========================================

# Construir URLs de exportación CSV
def get_url(gid):
    return f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={gid}"

# --- FUNCIÓN DE CARGA Y LIMPIEZA ---
@st.cache_data(ttl=600)
def load_data():
    urls = {
        'Android': get_url(GID_ANDROID),
        'iOS': get_url(GID_IOS)
    }
    
    dfs = []
    
    for os_name, url in urls.items():
        try:
            # Leer CSV
            df = pd.read_csv(url)
            
            # Limpiar nombres de columnas (quitar espacios extra y comillas)
            df.columns = df.columns.str.strip().str.replace('"', '')
            
            # --- NORMALIZAR NOMBRES DE COLUMNAS ---
            # Esto busca tus columnas aunque cambies mayúsculas/minúsculas
            cols_map = {col: col.lower() for col in df.columns}
            
            for col, lower_col in cols_map.items():
                if lower_col in ['date', 'day', 'fecha', 'v', 'día']:
                    df.rename(columns={col: 'Date'}, inplace=True)
                elif lower_col in ['cost', 'coste', 'spend', 'costo']:
                    df.rename(columns={col: 'Cost'}, inplace=True)
                elif lower_col in ['revenue', 'revenue total', 'ingresos', 'gain']: # Ojo, Gain suele ser beneficio, Revenue es ingreso bruto. Ajusta si tu columna se llama diferente.
                    df.rename(columns={col: 'Revenue'}, inplace=True)
                elif lower_col in ['country', 'pais', 'geo', 'geo/os']:
                    df.rename(columns={col: 'Country'}, inplace=True)

            # Validar que existan las columnas críticas
            required_cols = ['Date', 'Cost', 'Revenue', 'Country']
            # Nota: Si tu Excel no tiene columna 'Revenue', el código fallará. 
            # Si 'Gain' es tu única columna de beneficio, avísame para cambiarlo.
            
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                # Si falta Revenue pero hay Gain, intentamos usar Gain
                if 'Revenue' in missing and 'Gain' in df.columns:
                     df.rename(columns={'Gain': 'Revenue'}, inplace=True)
                else:
                    st.warning(f"⚠️ En la pestaña {os_name} faltan columnas: {missing}. Columnas detectadas: {list(df.columns)}")
                    continue

            # Limpiar datos numéricos ($ y ,)
            for col in ['Cost', 'Revenue']:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Convertir fecha
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Añadir etiqueta de Sistema Operativo
            df['OS'] = os_name
            
            dfs.append(df)
            
        except Exception as e:
            st.error(f"Error cargando pestaña {os_name}: {e}")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

# --- CARGAR DATOS ---
df = load_data()

# --- INTERFAZ ---
st.title("📊 Dashboard Financiero: Coste vs Revenue")

if df is not None and not df.empty:
    # ---------------------------
    # 1. SIDEBAR Y FILTROS
    # ---------------------------
    st.sidebar.header("Filtros")
    
    # Filtro Fecha
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    if pd.isnull(min_date) or pd.isnull(max_date):
        st.warning("No se detectaron fechas válidas.")
        date_range = [pd.Timestamp.today(), pd.Timestamp.today()]
    else:
        date_range = st.sidebar.date_input("Rango de Fechas", [min_date, max_date])
    
    # Filtro OS
    selected_os = st.sidebar.multiselect("Sistema Operativo", df['OS'].unique(), default=df['OS'].unique())
    
    # Filtro País
    all_countries = sorted(df['Country'].unique().astype(str))
    selected_countries = st.sidebar.multiselect("Países", all_countries, default=all_countries)
    
    # APLICAR FILTROS
    if len(date_range) == 2:
        mask = (
            (df['Date'] >= pd.to_datetime(date_range[0])) & 
            (df['Date'] <= pd.to_datetime(date_range[1])) &
            (df['OS'].isin(selected_os)) &
            (df['Country'].isin(selected_countries))
        )
        df_filtered = df[mask]
    else:
        df_filtered = df

    # ---------------------------
    # 2. KPIS GLOBALES
    # ---------------------------
    total_cost = df_filtered['Cost'].sum()
    total_rev = df_filtered['Revenue'].sum()
    total_profit = total_rev - total_cost # Asumiendo que Revenue es bruto. Si es neto, ajusta.
    roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💸 Coste Total", f"${total_cost:,.2f}")
    k2.metric("💰 Revenue Total", f"${total_rev:,.2f}")
    k3.metric("📈 Beneficio", f"${total_profit:,.2f}", delta_color="normal")
    k4.metric("🚀 ROI", f"{roi:.2f}%")
    
    st.divider()

    # ---------------------------
    # 3. GRÁFICAS SOLICITADAS
    # ---------------------------
    
    # A) EVOLUCIÓN DIARIA (TOTAL COSTE Y REVENUE)
    st.subheader("📅 1. Evolución Diaria (Total)")
    
    # Agrupar por Día (Sumando ambos OS)
    df_daily_total = df_filtered.groupby('Date')[['Cost', 'Revenue']].sum().reset_index()
    
    # Transformar para gráfico
    df_daily_melt = df_daily_total.melt(id_vars='Date', value_vars=['Cost', 'Revenue'], var_name='Metric', value_name='Amount')
    
    fig_daily = px.line(
        df_daily_melt, 
        x='Date', 
        y='Amount', 
        color='Metric', 
        title="Coste vs Revenue Diario (Global)",
        color_discrete_map={'Cost': '#EF553B', 'Revenue': '#00CC96'},
        markers=True
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    # B) DETALLE POR PAÍS Y DÍA (Separado por OS)
    st.subheader("🌍 2. Rendimiento por País y Día (Separado por OS)")
    
    # Esta gráfica puede ser muy densa si hay muchos días.
    # Hacemos un gráfico de barras apiladas o agrupadas.
    
    # Opción: Elegir una métrica para visualizar
    metric_to_plot = st.radio("Selecciona métrica para el gráfico detallado:", ["Revenue", "Cost"], horizontal=True)
    
    color_map = {'Android': '#3DDC84', 'iOS': '#000000'} # Colores típicos
    
    fig_country_day = px.bar(
        df_filtered,
        x='Date',
        y=metric_to_plot,
        color='OS',
        facet_col='Country', # Una gráfica pequeña por país
        facet_col_wrap=3,    # Máximo 3 gráficas por fila
        title=f"Evolución de {metric_to_plot} por País y Sistema Operativo",
        color_discrete_map=color_map,
        height=800 # Hacemos la gráfica más alta para que quepan
    )
    # Ajustar ejes para que se vean bien las fechas
    fig_country_day.update_xaxes(matches=None) 
    st.plotly_chart(fig_country_day, use_container_width=True)

    # C) TABLA RESUMEN POR PAÍS
    st.subheader("📋 Resumen por País")
    df_summary = df_filtered.groupby(['Country', 'OS'])[['Cost', 'Revenue']].sum().reset_index()
    df_summary['ROI %'] = ((df_summary['Revenue'] - df_summary['Cost']) / df_summary['Cost'] * 100).fillna(0)
    
    # Formatear columnas
    st.dataframe(
        df_summary.style.format({
            "Cost": "${:,.2f}", 
            "Revenue": "${:,.2f}", 
            "ROI %": "{:.1f}%"
        }), 
        use_container_width=True
    )

    with st.expander("📂 Ver Datos Brutos Completos"):
        st.dataframe(df_filtered)

else:
    st.info("👋 Conectando... Si esto tarda, revisa que el Excel esté compartido como 'Cualquiera con el enlace'.")
