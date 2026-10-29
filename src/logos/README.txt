Logos for the viewer's title bar (viewer_template.html). make_viewer_index.py copies this
folder next to viewer.html, so the page loads them as logos/<name>.

  tamu.png   Texas A&M University logo       (shown first, links to tamu.edu)
  atmo.png   Department of Atmospheric Sciences logo (links to the department page)

Use white / reversed versions; the bar is maroon. They are drawn 28 px tall (24 px on
phones, where only the first one is shown), so a file about 60-120 px tall stays sharp.
SVG works too: rename the src in viewer_template.html. A missing file is simply hidden.
