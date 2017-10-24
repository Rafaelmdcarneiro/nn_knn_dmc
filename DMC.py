import sklearn
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

import numpy as np
from sklearn import datasets
from sklearn.neighbors import NearestCentroid

n_neighbors = 15

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

h = .02  # step size in the mesh

cmap_light =[[0, '#FFAAAA'], [0.5, '#AAFFAA'], [1, '#AAAAFF']]
cmap_bold = [[0, '#FF0000'], [0.5, '#00FF00'], [1, '#0000FF']]

data = []
titles = []
i = 0

for shrinkage in [None, .2]:
    clf = NearestCentroid(shrink_threshold=shrinkage)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print(shrinkage, np.mean(y == y_pred))
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_ = np.arange(x_min, x_max, h)
    y_ = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(x_, y_)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    data.append([])
    p1 = go.Heatmap(x=x_, y=y_, z=Z,
                    showscale=False,
                    colorscale=cmap_light)
    
    p2 = go.Scatter(x=X[:, 0], y=X[:, 1], 
                    mode='markers',
                    marker=dict(color=X[:, 0],
                                colorscale=cmap_bold,
                                line=dict(color='black', width=1)))
    data[i].append(p1)
    data[i].append(p2) 
    titles.append("3-Class classification (shrink_threshold=%r)"
                   % shrinkage)
    i+=1

fig = tools.make_subplots(rows=1, cols=2,
                          subplot_titles=tuple(titles), 
                          print_grid=False)

for i in range(0, len(data)):
    for j in range(0, len(data[i])):
        fig.append_trace(data[i][j], 1, i+1)

fig['layout'].update(height=700, hovermode='closest', 
                     showlegend=False)

                     
py.iplot(fig)