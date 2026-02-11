import streamlit as st
import pandas as pd
import plotly.express as px
import time

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Dashboard de Tráfico Diario",
    page_icon="📊",
    layout="wide"
)

# --- TU ENLACE MÁGICO ---
# Aquí pegas el enlace CSV de tu Google Sheet o el enlace RAW de GitHub/Dropbox
# Ejemplo (este es falso, pon el tuyo):
DATA_URL = "https://docs.google.com/spreadsheets/d/e/TuIDdeGoogleSheet/pub?output=csv"

# --- FUNCIÓN PARA CARGAR DATOS ---
# El 'ttl=600' significa que guarda los datos en memoria 10 minutos para ir rápido.
# Pasado ese tiempo, vuelve a mirar el Excel para ver si hay cambios.
@st.cache_data(ttl=600)
def load_data(url):
    try:
        # Leemos el CSV
        df = pd.read_csv(url)
        
        # LIMPIEZA: Quitamos comillas y espacios de los nombres de columnas
        df.columns = df.columns.str.strip().str.replace('"', '')
        
        # LIMPIEZA: Aseguramos que los números sean números
        # A veces el CSV trae simbolos de moneda o comas extrañas
        cols_to_clean = ['Cost', 'Impressions']
        for col in cols_to_clean:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        return None

# --- INTERFAZ PRINCIPAL ---
st.title("🚀 Dashboard de Rendimiento Diario")
st.markdown(f"**Fuente de datos:** {DATA_URL}")

# Botón para forzar actualización manual
if st.button('🔄 Actualizar Datos Ahora'):
    st.cache_data.clear()
    st.rerun()

# Cargamos los datos
df = load_data(DATA_URL)

if df is not None:
    # --- FILTROS LATERALES ---
    st.sidebar.header("Filtros")
    
    # Filtro por País (si la columna existe)
    if 'Country' in df.columns:
        countries = st.sidebar.multiselect(
            "Selecciona Países:",
            options=df['Country'].unique(),
            default=df['Country'].unique()
        )
        df_filtered = df[df['Country'].isin(countries)]
    else:
        df_filtered = df

    # --- KPIs (INDICADORES CLAVE) ---
    total_cost = df_filtered['Cost'].sum()
    total_imps = df_filtered['Impressions'].sum()
    # Calculamos CPM Promedio (Coste / (Impresiones/1000))
    avg_cpm = (total_cost / (total_imps / 1000)) if total_imps > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Gasto Total", f"${total_cost:,.2f}")
    col2.metric("👁️ Impresiones Totales", f"{total_imps:,.0f}")
    col3.metric("📉 CPM Promedio", f"${avg_cpm:.2f}")

    st.markdown("---")

    # --- GRÁFICOS ---
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Gasto por País")
        fig_cost = px.bar(
            df_filtered, 
            x='Country', 
            y='Cost',
            text='Cost',
            color='Cost',
            color_continuous_scale='Reds',
            title="Inversión ($) desglosada"
        )
        fig_cost.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        st.plotly_chart(fig_cost, use_container_width=True)

    with col_chart2:
        st.subheader("Relación Volumen vs Coste")
        # Gráfico de dispersión para ver eficiencia
        fig_scatter = px.scatter(
            df_filtered,
            x='Impressions',
            y='Cost',
            size='Cost',
            color='Country',
            hover_name='Country',
            title="¿Quién da más tráfico por menos dinero?"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- TABLA DE DATOS ---
    with st.expander("Ver Tabla de Datos Completa"):
        st.dataframe(df_filtered)

else:
    st.error("⚠️ No se pudieron cargar los datos. Revisa que el enlace al Excel/CSV sea público y correcto.")