from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SearchField, SubmitField
from wtforms.validators import DataRequired
from services.similarity_search import data_query
import webbrowser
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Hello, world'

class SearchBar(FlaskForm):
    search = StringField("Lookup an image", [DataRequired()])
    submit = SubmitField('Search')
    # num = IntegerField("Lookup an image")


@app.route("/Home", methods=['GET', 'POST'])
def HomePage():
    search = None
    form = SearchBar()
    images = None
    if form.validate_on_submit():
        search = form.search.data
        form.search.data = ''
    if search is None:
        pass
    else:
        images = data_query(search)
    
    return render_template("home.html", search=search, images = images, form = form)



if __name__ == "__main__":
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new("http://127.0.0.1:5000/Home")
    app.run(port=5000, debug=True)