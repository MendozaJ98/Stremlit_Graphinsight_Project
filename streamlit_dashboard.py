import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np
# Establecer el tema para los gráficos de Plotly
px.defaults.template = "plotly_dark"

# Desactivar el mensaje de advertencia
st.set_option('deprecation.showPyplotGlobalUse', False)

# Crear una aplicación de Streamlit
st.set_page_config(layout="wide")
st.title("GraphInsight")

import numpy as np

def generate_relational_graph(df, columns):
    fig = go.Figure()

    unique_categories = set()
    node_index = {}

    # Assign an index to each unique category
    for column in columns:
        unique_categories |= set(df[column].unique())

    for i, category in enumerate(unique_categories):
        node_index[category] = i

    processed_combinations = set()  # Track processed combinations of categories

    # Convert relevant columns to numpy arrays
    category_arrays = [np.array(df[column]) for column in columns]

    for i in range(len(df)):
        source_indices = [node_index[category_array[i]] for category_array in category_arrays]

        # Check if combination has already been processed
        combination = tuple(source_indices)
        if combination in processed_combinations:
            continue

        processed_combinations.add(combination)

        fig.add_trace(go.Scatter(x=source_indices, y=list(range(len(columns))),
                                 mode='lines',
                                 line=dict(width=2),
                                 hoverinfo='none'))

        for j, index in enumerate(source_indices):
            fig.add_trace(go.Scatter(x=[index], y=[j],
                                     mode='markers',
                                     marker=dict(symbol='circle', size=10),
                                     name=str(category_arrays[j][i])))

    fig.update_layout(title="Gráfico de Relaciones", showlegend=False, hovermode='closest')

    # Set x-axis tick labels
    x_tick_labels = list(node_index.keys())
    x_tick_vals = list(node_index.values())
    fig.update_xaxes(ticktext=x_tick_labels, tickvals=x_tick_vals)

    # Display variable descriptions
    st.subheader("Descripción de Variables")
    variable_descriptions = {}
    for column in columns:
        variable_descriptions[column] = df[column].describe().to_string()
    st.write(variable_descriptions)

    # Display the graph
    st.plotly_chart(fig)




def main():
    # Cargar un archivo CSV o Excel
    file = st.file_uploader("Cargar un archivo CSV o Excel", type=["csv", "xlsx"])

    if file is not None:
        # Leer el archivo cargado en un DataFrame
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(file)

        # Mostrar el DataFrame
        st.subheader("Datos")
        st.dataframe(df)

        # Generar opciones para que el usuario seleccione variables y tipos de gráficos
        variables = list(df.columns)
        variables.insert(0, "Seleccionar una variable")

        # Seleccionar variables para Sugerencias de Visualización
        selected_variables = st.multiselect("Seleccionar variables", variables)
        if len(selected_variables) > 0:
            suggestions = px.scatter_matrix(df[selected_variables])
        else:
            suggestions = None

        # Sugerencias de Visualización
        with st.expander("Matriz de Dispersión"):
            if suggestions is not None:
                st.plotly_chart(suggestions, use_container_width=True, height=800)
            else:
                st.write("No se han seleccionado variables.")

        # Mapa de Calor de Correlación
        with st.expander("Mapa de Calor de Correlación"):
            numeric_columns = df.select_dtypes(include=[float, int]).columns
            corr_matrix = df[numeric_columns].corr()
            fig = ff.create_annotated_heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.index.tolist(),
                colorscale=["blue", "red"],  # Colormap personalizado
                annotation_text=corr_matrix.round(2).values,
                showscale=True,
            )
            fig.update_layout(width=800, height=500, title="Mapa de Calor de Correlación")
            st.plotly_chart(fig)

        # Gráfico de Distribución
        with st.expander("Gráfico de Distribución"):
            numeric_columns = df.select_dtypes(include=[float, int]).columns
            selected_column = st.selectbox("Seleccionar una columna numérica para el gráfico de distribución",
                                           numeric_columns)
            fig = px.histogram(df, x=selected_column, nbins=30)
            fig.update_layout(width=800, height=500, title="Gráfico de Distribución")
            st.plotly_chart(fig)

        # Gráficos Personalizados
        with st.expander("Gráfico Personalizado"):
            x_col = st.selectbox("Seleccionar variable para el eje X", variables)
            y_col = st.selectbox("Seleccionar variable para el eje Y", variables)
            graph_type = st.selectbox("Seleccionar Tipo de Gráfico",
                                      ["Dispersión", "Linea", "Barras", "Histograma", "Box", "Pastel", "Sunburst",
                                       "Contorno de Densidad"])

            if st.button("Generar Gráfico"):
                if x_col != "Seleccionar una variable" and y_col != "Seleccionar una variable":
                    if graph_type == "Dispersión":
                        fig = px.scatter(df, x=x_col, y=y_col)
                    elif graph_type == "Linea":
                        fig = px.line(df, x=x_col, y=y_col)
                    elif graph_type == "Barras":
                        fig = px.bar(df, x=x_col, y=y_col)
                    elif graph_type == "Histograma":
                        fig = px.histogram(df, x=x_col)
                    elif graph_type == "Box":
                        fig = px.box(df, x=x_col, y=y_col)
                    elif graph_type == "Pastel":
                        fig = px.pie(df, names=x_col, values=y_col)
                    elif graph_type == "Sunburst":
                        fig = px.sunburst(df, path=[x_col, y_col])
                    elif graph_type == "Contorno de Densidad":
                        fig = px.density_contour(df, x=x_col, y=y_col)

                    fig.update_layout(width=800, height=500, title="Gráfico Personalizado")
                    st.plotly_chart(fig)
                else:
                    st.write("Por favor, selecciona variables para los ejes X e Y.")

        # Líneas de Tiempo
        with st.expander("Líneas de Tiempo"):
            date_columns = df.select_dtypes(include=["datetime"]).columns
            selected_date_column = st.selectbox("Seleccionar una columna de fecha para la línea de tiempo",
                                                date_columns)
            group_by_variable = st.selectbox("Seleccionar una variable para agrupar", variables[1:])

            if st.button("Generar Línea de Tiempo"):
                if selected_date_column is not None and group_by_variable is not None:
                    df["Mes"] = pd.to_datetime(df[selected_date_column]).dt.to_period('M').astype(str)
                    timeline_df = df.groupby(["Mes", group_by_variable]).size().reset_index(name="Conteo")
                    fig = px.line(timeline_df, x="Mes", y="Conteo", color=group_by_variable)
                    fig.update_layout(width=800, height=500, title="Línea de Tiempo")
                    st.plotly_chart(fig)
                else:
                    st.write(
                        "No se ha seleccionado una columna de fecha o una variable para agrupar en la línea de tiempo.")

        # Gráfico de Relaciones
        with st.expander("Gráfico de Relaciones"):
            relational_columns = list(df.columns)
            relational_columns.insert(0, "Seleccionar una columna")
            selected_relational_columns = st.multiselect("Seleccionar columnas para el gráfico de relaciones", relational_columns, key="relational_columns")

            if len(selected_relational_columns) > 1 and "Seleccionar una columna" not in selected_relational_columns:
                # Generate the relational graph
                generate_relational_graph(df, selected_relational_columns)
            else:
                st.write("Por favor, selecciona al menos dos columnas para el gráfico de relaciones.")


if __name__ == "__main__":
    main()
