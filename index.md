---
title: Functional Normalization Toolbox
layout: default
---

This GitHub Project Page is used to keep my notes while developing the Python version of Functional Normalization Toolbox.
It contains some useful formulas and API information.


## Status

[![Build Status](https://travis-ci.org/feilong/funcnorm.svg?branch=rf-python)](https://travis-ci.org/feilong/funcnorm)
[![Coverage Status](https://coveralls.io/repos/feilong/funcnorm/badge.svg?branch=rf-python&service=github)](https://coveralls.io/github/feilong/funcnorm?branch=rf-python)


## Pages

{% for page in site.pages %}
{% if page.url != "/index.html" %}
[{{ page.title }}]({{ site.baseurl }}{{ page.url }})
{% endif %}
{% endfor %}
