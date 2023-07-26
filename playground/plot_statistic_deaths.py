import matplotlib.pyplot as plt
import numpy as np

plt.rcdefaults()
fig, ax = plt.subplots()
fig.set_figheight(4.5)
width = 0.35  # the width of the bars

disease = ('Chronic ischemic heart disease', 'Unspecified dementia', 'Lung and bronchial cancer',
           'Acute myocardial infarction', 'Heart insufficiency ', 'Other chronic obstructive\npulmonary disease',
           'Other imprecise or unspecified\ncauses of death', 'Hypertensive heart disease',
           'Atrial Fibrillation and atrial flutter', 'Malignant neoplasm of the pancreas')
y_pos = 0.75 * np.arange(len(disease))
values = [73.5, 45.1, 44.8, 44.3, 35.3, 31.4, 24.1, 22.0, 20.7, 19.2]
ax.set_xlim(0, 90)

ax.barh(y_pos, values, width, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(disease, multialignment='right')
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Number of deaths (in thousands)')

for i, v in enumerate(values):
    ax.text(v + 3, 0.75 * i + 0.07, str(v), fontweight='bold')

plt.tight_layout()
plt.savefig("plots/" + "TOP10_causes_death_Germany_2019.pdf")

# Highlight CVDs
CVD = ['Chronic ischemic heart disease', 'Acute myocardial infarction', 'Heart insufficiency ',
       'Hypertensive heart disease',
       'Atrial Fibrillation and atrial flutter']
for tl in ax.get_yticklabels():
    txt = tl.get_text()
    if txt in CVD:
        txt += ' (!)'
        tl.set_backgroundcolor('yellow')
    tl.set_text(txt)

plt.savefig("plots/" + "TOP10_causes_death_Germany_2019_highlighted.pdf")
plt.show()
