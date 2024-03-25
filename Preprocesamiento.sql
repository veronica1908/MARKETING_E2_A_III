{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOSSVKkHraTdeDQG4UXLSP7",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/veronica1908/MARKETING_E2_A_III/blob/main/Preprocesamiento.sql\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Instalar la extensión ipython-sql\n",
        "!pip install -q ipython-sql\n",
        "\n",
        "# Cargar la extensión\n",
        "%load_ext sql\n",
        "\n",
        "# Instalar bibliotecas necesarias\n",
        "!pip install sqlalchemy\n",
        "!pip install mysql-connector-python\n",
        "\n",
        "# Importar bibliotecas\n",
        "import sqlite3\n",
        "from sqlalchemy import create_engine"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "V0o615IrEUoi",
        "outputId": "d564de8f-59d2-433b-c71d-d2fa8c5d753e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The sql extension is already loaded. To reload it, use:\n",
            "  %reload_ext sql\n",
            "Requirement already satisfied: sqlalchemy in /usr/local/lib/python3.10/dist-packages (2.0.28)\n",
            "Requirement already satisfied: typing-extensions>=4.6.0 in /usr/local/lib/python3.10/dist-packages (from sqlalchemy) (4.10.0)\n",
            "Requirement already satisfied: greenlet!=0.4.17 in /usr/local/lib/python3.10/dist-packages (from sqlalchemy) (3.0.3)\n",
            "Collecting mysql-connector-python\n",
            "  Downloading mysql_connector_python-8.3.0-cp310-cp310-manylinux_2_17_x86_64.whl (21.5 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m21.5/21.5 MB\u001b[0m \u001b[31m45.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: mysql-connector-python\n",
            "Successfully installed mysql-connector-python-8.3.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Conectarse a la base de datos\n",
        "conn = sqlite3.connect('db_movies')"
      ],
      "metadata": {
        "id": "EUAoeyyoEYm5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Crear tabla con usuarios con más de 50 películas vistas y menos de 1000\n",
        "\n",
        "# Conectarse a la base de datos\n",
        "conn = sqlite3.connect('db_movies')\n",
        "\n",
        "# Crear tabla con usuarios con más de 50 películas vistas y menos de 1000\n",
        "query = \"\"\"\n",
        "    DROP TABLE IF EXISTS usuarios_sel;\n",
        "    CREATE TABLE usuarios_sel AS\n",
        "    SELECT \"userId\" AS user_id, COUNT(*) AS cnt_rat\n",
        "    FROM ratings\n",
        "    GROUP BY \"userId\"\n",
        "    HAVING cnt_rat > 50 AND cnt_rat <= 1000\n",
        "    ORDER BY cnt_rat DESC;\n",
        "\"\"\"\n",
        "\n",
        "# Ejecutar la consulta\n",
        "conn.executescript(query)\n",
        "\n",
        "# Cerrar la conexión\n",
        "conn.close()"
      ],
      "metadata": {
        "id": "8ZkUqUGXDK7R"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Conectarse a la base de datos\n",
        "conn = sqlite3.connect('db_movies')\n",
        "\n",
        "# Crear tabla con películas que han sido calificadas por más de 50 usuarios\n",
        "query = \"\"\"\n",
        "DROP TABLE IF EXISTS movies_sel;\n",
        "\n",
        "CREATE TABLE movies_sel AS\n",
        "SELECT movieId,\n",
        "       COUNT(*) AS cnt_rat\n",
        "FROM ratings\n",
        "GROUP BY movieId\n",
        "HAVING cnt_rat > 50\n",
        "ORDER BY cnt_rat DESC;\n",
        "\"\"\"\n",
        "\n",
        "# Ejecutar la consulta\n",
        "conn.executescript(query)\n",
        "\n",
        "# Cerrar la conexión\n",
        "conn.close()"
      ],
      "metadata": {
        "id": "vYY1g_MjFvN9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Conectarse a la base de datos\n",
        "conn = sqlite3.connect('db_movies')\n",
        "\n",
        "# Crear tablas filtradas de calificaciones con usuarios y películas\n",
        "##ratings_final\n",
        "\n",
        "query = \"\"\"\n",
        "DROP TABLE IF EXISTS ratings_final;\n",
        "\n",
        "CREATE TABLE ratings_final AS\n",
        "SELECT a.\"userId\" AS user_id,\n",
        "       a.movieId AS movie_id,\n",
        "       a.rating AS rating\n",
        "FROM ratings a\n",
        "INNER JOIN movies_sel b ON a.movieId = b.movieId\n",
        "INNER JOIN usuarios_sel c ON a.\"userId\" = c.user_id;\n",
        "\"\"\"\n",
        "\n",
        "##movies_final\n",
        "\"\"\"\n",
        "DROP TABLE IF EXISTS movies_final;\n",
        "\n",
        "CREATE TABLE movies_final AS\n",
        "SELECT a.movieId AS movie_id,\n",
        "       a.title AS movie_title,\n",
        "       a.genres AS genres\n",
        "FROM movies a\n",
        "INNER JOIN movies_sel c ON a.movieId = c.movieId;\n",
        "\"\"\"\n",
        "\n",
        "# Ejecutar la consulta\n",
        "conn.executescript(query)\n",
        "\n",
        "# Cerrar la conexión\n",
        "conn.close()"
      ],
      "metadata": {
        "id": "77ryWF1UGbX5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7GVBstjRCvnL",
        "outputId": "8efb55da-a6ef-4ac6-9a04-40544f56a657"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Tabla 'full_ratings' creada con éxito.\n"
          ]
        }
      ],
      "source": [
        "# Conectarse a la base de datos\n",
        "conn = sqlite3.connect('db_movies')\n",
        "cursor = conn.cursor()\n",
        "\n",
        "# Script SQL\n",
        "script = \"\"\"\n",
        "DROP TABLE IF EXISTS full_ratings;\n",
        "\n",
        "CREATE TABLE full_ratings AS\n",
        "SELECT a.*,\n",
        "       c.movie_title,\n",
        "       c.genres\n",
        "FROM ratings_final a\n",
        "INNER JOIN movies_final c ON a.movie_id = c.movie_id;\n",
        "\"\"\"\n",
        "\n",
        "# Ejecutar el script SQL\n",
        "cursor.executescript(script)\n",
        "\n",
        "# Confirmar los cambios y cerrar la conexión\n",
        "conn.commit()\n",
        "conn.close()\n",
        "\n",
        "print(\"Tabla 'full_ratings' creada con éxito.\")"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Exportar la tabla preprocesada en SQL"
      ],
      "metadata": {
        "id": "HbK4YD00KKLJ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Conectar a la base de datos\n",
        "conn = sqlite3.connect('db_movies')\n",
        "\n",
        "# Consulta SQL para seleccionar todos los datos de la tabla full_ratings\n",
        "query = \"SELECT * FROM full_ratings\"\n",
        "\n",
        "# Leer los datos en un DataFrame de pandas\n",
        "df = pd.read_sql_query(query, conn)\n",
        "\n",
        "# Guardar el DataFrame como un archivo CSV\n",
        "df.to_csv('full_ratings.csv', index=False)\n",
        "\n",
        "# Cerrar la conexión\n",
        "conn.close()\n",
        "\n",
        "print(\"La tabla 'full_ratings' se ha exportado como full_ratings.csv\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "d-n-CdNLKRCY",
        "outputId": "4d855057-4e47-4cad-9944-f5a58324c087"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "La tabla 'full_ratings' se ha exportado como full_ratings.csv\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import files\n",
        "\n",
        "files.download('full_ratings.csv')\n",
        "##Esta tabla se utilizará en la fase de modelos."
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 17
        },
        "id": "ff9jWWq8KeAo",
        "outputId": "7ca215f4-f9a9-432b-f96a-3aa8c1efd625"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "\n",
              "    async function download(id, filename, size) {\n",
              "      if (!google.colab.kernel.accessAllowed) {\n",
              "        return;\n",
              "      }\n",
              "      const div = document.createElement('div');\n",
              "      const label = document.createElement('label');\n",
              "      label.textContent = `Downloading \"${filename}\": `;\n",
              "      div.appendChild(label);\n",
              "      const progress = document.createElement('progress');\n",
              "      progress.max = size;\n",
              "      div.appendChild(progress);\n",
              "      document.body.appendChild(div);\n",
              "\n",
              "      const buffers = [];\n",
              "      let downloaded = 0;\n",
              "\n",
              "      const channel = await google.colab.kernel.comms.open(id);\n",
              "      // Send a message to notify the kernel that we're ready.\n",
              "      channel.send({})\n",
              "\n",
              "      for await (const message of channel.messages) {\n",
              "        // Send a message to notify the kernel that we're ready.\n",
              "        channel.send({})\n",
              "        if (message.buffers) {\n",
              "          for (const buffer of message.buffers) {\n",
              "            buffers.push(buffer);\n",
              "            downloaded += buffer.byteLength;\n",
              "            progress.value = downloaded;\n",
              "          }\n",
              "        }\n",
              "      }\n",
              "      const blob = new Blob(buffers, {type: 'application/binary'});\n",
              "      const a = document.createElement('a');\n",
              "      a.href = window.URL.createObjectURL(blob);\n",
              "      a.download = filename;\n",
              "      div.appendChild(a);\n",
              "      a.click();\n",
              "      div.remove();\n",
              "    }\n",
              "  "
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.Javascript object>"
            ],
            "application/javascript": [
              "download(\"download_6e040760-506b-4b6b-8ae6-2578dd67a38c\", \"full_ratings.csv\", 2009297)"
            ]
          },
          "metadata": {}
        }
      ]
    }
  ]
}