"""
Sales Plugin para Semantic Kernel.

En SK las tools se agrupan en clases llamadas Plugins.
Cada método con @kernel_function es una tool que el LLM puede invocar.

Equivalencia con LangGraph:
  @tool def tool_ventas_por_vendedor()  →  @kernel_function def ventas_por_vendedor()
  La docstring es lo que el LLM lee para decidir cuándo usar cada función.
"""

import json
from semantic_kernel.functions import kernel_function

from app.tools import (
    ventas_por_vendedor,
    ventas_por_categoria,
    ventas_por_region,
    ventas_por_mes,
    ventas_vendedor_por_mes,
    ventas_producto_por_region,
    lista_productos,
    producto_mas_vendido,
    resumen_general,
)


class SalesPlugin:
    """Plugin con todas las herramientas de análisis de ventas."""

    @kernel_function(
        name="ventas_por_vendedor",
        description="Obtiene el total de ventas en pesos agrupado por vendedor. Útil para comparar rendimiento del equipo comercial."
    )
    def get_ventas_por_vendedor(self) -> str:
        return json.dumps(ventas_por_vendedor(), ensure_ascii=False)

    @kernel_function(
        name="ventas_por_categoria",
        description="Obtiene el total de ventas agrupado por categoría de producto."
    )
    def get_ventas_por_categoria(self) -> str:
        return json.dumps(ventas_por_categoria(), ensure_ascii=False)

    @kernel_function(
        name="ventas_por_region",
        description="Obtiene el total de ventas agrupado por región geográfica (Norte, Sur, Centro)."
    )
    def get_ventas_por_region(self) -> str:
        return json.dumps(ventas_por_region(), ensure_ascii=False)

    @kernel_function(
        name="ventas_por_mes",
        description="Obtiene la evolución de ventas mes a mes. Útil para detectar tendencias temporales."
    )
    def get_ventas_por_mes(self) -> str:
        return json.dumps(ventas_por_mes(), ensure_ascii=False)

    @kernel_function(
        name="ventas_vendedor_por_mes",
        description="Obtiene las ventas mes a mes de un vendedor específico. Usar cuando pregunten cuánto vendió una persona en un mes."
    )
    def get_ventas_vendedor_por_mes(self, vendedor: str) -> str:
        return json.dumps(ventas_vendedor_por_mes(vendedor), ensure_ascii=False)

    @kernel_function(
        name="ventas_por_producto",
        description="Obtiene en qué regiones se vende un producto específico con unidades y pesos por región."
    )
    def get_ventas_por_producto(self, producto: str) -> str:
        return json.dumps(ventas_producto_por_region(producto), ensure_ascii=False)

    @kernel_function(
        name="lista_productos",
        description="Lista todos los productos que vende la tienda con nombre, categoría y unidades vendidas."
    )
    def get_lista_productos(self) -> str:
        return json.dumps(lista_productos(), ensure_ascii=False)

    @kernel_function(
        name="producto_mas_vendido",
        description="Obtiene el producto con mayor cantidad de unidades vendidas en todo el período."
    )
    def get_producto_mas_vendido(self) -> str:
        return json.dumps(producto_mas_vendido(), ensure_ascii=False)

    @kernel_function(
        name="resumen_general",
        description="Obtiene un resumen ejecutivo con las métricas principales: ingresos totales, ticket promedio, cantidad de transacciones."
    )
    def get_resumen_general(self) -> str:
        return json.dumps(resumen_general(), ensure_ascii=False)
