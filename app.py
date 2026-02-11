import streamlit as st
import pandas as pd
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Dashboard Master: Cost vs Revenue", page_icon="💰", layout="wide")

# ==========================================
# ⚙️ CONFIGURACIÓN DE DATOS
# ==========================================
SHEET_ID = "1WuBv1esTxZAfC07BPwWzjsz5TZqfUHa6MOzNIAEOMew"
GID_ANDROID = "368085162"    # Datos por AOS/dia
GID_IOS = "1225911759"       # Datos por IOS/dia

def get_url(gid):
    return f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={gid}"

# --- FUNCIÓN DE LIMPIEZA INTELIGENTE ---
def find_column(columns, candidates):
    """Busca la primera coincidencia de una lista de candidatos en las columnas."""
    # Normalizamos columnas del excel a minúsculas
    cols_lower = [c.lower() for c in columns]
    
    for candidate in candidates:
        if candidate in cols_lower:
            # Devolvemos el nombre real de la columna (con mayúsculas originales)
            return columns[cols_lower.index(candidate)]
    return None

@st.cache_data(ttl=600)
def load_data():
    urls = {'Android': get_url(GID_ANDROID), 'iOS': get_url(GID_IOS)}
    dfs = []
    
    for os_name, url in urls.items():
        try:
            # 1. Leer CSV
            df = pd.read_csv(url)
            
            # 2. Limpieza básica de cabeceras
            df.columns = df.columns.str.strip().str.replace('"', '')
            
            # 3. MAPEO SELECTIVO (Evita duplicados)
            # Buscamos la columna exacta para cada métrica
            col_mapping = {}
            
            # Prioridad para FECHA: 'day', 'date', 'fecha', 'v'
            real_date_col = find_column(df.columns, ['day', 'date', 'fecha', 'v', 'time'])
            if real_date_col: col_mapping[real_date_col] = 'Date'
            
            # Prioridad para PAÍS: 'country', 'geo', 'geo/os', 'pais'
            real_country_col = find_column(df.columns, ['country', 'geo', 'geo/os', 'pais'])
            if real_country_col: col_mapping[real_country_col] = 'Country'
            
            # Prioridad para COSTE: 'cost', 'coste', 'spend'
            real_cost_col = find_column(df.columns, ['cost', 'coste', 'spend', 'total cost'])
            if real_cost_col: col_mapping[real_cost_col] = 'Cost'
            
            # Prioridad para REVENUE: 'revenue total' > 'revenue' > 'gain'
            # (Aquí estaba el fallo: ahora prioriza 'Revenue Total' y ignora 'Gain' si ya encontró el otro)
            real_rev_col = find_column(df.columns, ['revenue total', 'revenue', 'ingresos', 'gain'])
            if real_rev_col: col_mapping[real_rev_col] = 'Revenue'

            # 4. Renombrar solo las encontradas
            df.rename(columns=col_mapping, inplace=True)
            
            # 5. ELIMINAR DUPLICADOS (El salvavidas 🚑)
            # Si por error hay dos columnas 'Date', nos quedamos solo con la primera
            df = df.loc[:, ~df.columns.duplicated()]

            # 6. Validar columnas críticas
            required = ['Date', 'Cost', 'Revenue', 'Country']
            missing = [c for c in required if c not in df.columns]
            
            if missing:
                st.warning(f"⚠️ Pestaña {os_name}: Faltan columnas {missing}. Se encontraron: {list(df.columns)}")
                continue # Saltamos esta pestaña si está rota

            # 7. Limpiar números ($ y ,)
            for col in ['Cost', 'Revenue']:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # 8. Convertir fecha
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['OS'] = os_name
            
            # Seleccionamos solo las columnas que nos interesan para limpiar el DF final
            final_cols = ['Date', 'Country', 'Cost', 'Revenue', 'OS']
            dfs.append(df[final_cols])
            
        except Exception as e:
            st.error(f"Error procesando {os_name}: {e}")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

# --- CARGAR ---
df = load_data()

# --- INTERFAZ ---
st.title("📊 Dashboard Financiero: Coste vs Revenue")

if df is not None and not df.empty:
    # FILTROS
    st.sidebar.header("Filtros")
    
    min_date, max_date = df['Date'].min(), df['Date'].max()
    date_range = st.sidebar.date_input("Fechas", [min_date, max_date]) if not pd.isnull(min_date) else [None, None]
    
    selected_os = st.sidebar.multiselect("OS", df['OS'].unique(), default=df['OS'].unique())
    all_countries = sorted(df['Country'].unique().astype(str))
    selected_countries = st.sidebar.multiselect("Países", all_countries, default=all_countries)
    
    # FILTRADO
    if len(date_range) == 2 and date_range[0]:
        mask = (
            (df['Date'] >= pd.to_datetime(date_range[0])) & 
            (df['Date'] <= pd.to_datetime(date_range[1])) &
            (df['OS'].isin(selected_os)) &
            (df['Country'].isin(selected_countries))
        )
        df_filtered = df[mask]
    else:
        df_filtered = df

    # KPIS
    cost = df_filtered['Cost'].sum()
    rev = df_filtered['Revenue'].sum()
    profit = rev - cost
    roi = (profit / cost * 100) if cost > 0 else 0
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Coste", f"${cost:,.2f}")
    k2.metric("Revenue", f"${rev:,.2f}")
    k3.metric("Beneficio", f"${profit:,.2f}")
    k4.metric("ROI", f"{roi:.2f}%")
    
    st.divider()

    # GRÁFICA 1: EVOLUCIÓN
    st.subheader("📅 Evolución Diaria (Total)")
    daily = df_filtered.groupby('Date')[['Cost', 'Revenue']].sum().reset_index().melt(id_vars='Date')
    st.plotly_chart(px.line(daily, x='Date', y='value', color='variable', 
                            color_discrete_map={'Cost':'#EF553B', 'Revenue':'#00CC96'}), use_container_width=True)

    # GRÁFICA 2: PAÍS Y OS
    st.subheader("🌍 Rendimiento por País y OS")
    # Agrupamos para simplificar
    country_os = df_filtered.groupby(['Country', 'OS'])[['Cost', 'Revenue']].sum().reset_index().melt(id_vars=['Country', 'OS'])
    st.plotly_chart(px.bar(country_os, x='Country', y='value', color='variable', facet_col='OS', barmode='group',
                           color_discrete_map={'Cost':'#EF553B', 'Revenue':'#00CC96'}), use_container_width=True)

    # TABLA
    with st.expander("Ver Datos"):
        st.dataframe(df_filtered)

else:
    st.info("Cargando datos... Si ves esto mucho tiempo, revisa los permisos del Sheet.")
