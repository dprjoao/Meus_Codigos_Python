# Import packages
from dash import Dash, html

# Initialize the app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.Div(children='Olá Mundo')
])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)