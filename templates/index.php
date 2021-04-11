<html>

<head>
    <title>Face Grouper</title>

    <!-- stylesheets -->
    <link href="https://fonts.gstatic.com" rel="preconnect">
    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <link href="static/images/keyboard-icon.png" type="image/png" rel="icon"/>

    <!-- custom stylesheets -->
    <link href="static/css/style.css" type="text/css" rel="stylesheet">

    <!-- scripts -->
    <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/chart.js@2.9.4/dist/Chart.min.js"></script>
    <script type="text/javascript" src="//code.jquery.com/jquery-3.3.1.min.js"></script>
    <script type="text/javascript" src="//cdnjs.cloudflare.com/ajax/libs/socket.io/1.3.6/socket.io.min.js"></script>

    <!-- custom scripts -->
    <script type="text/javascript" src="https://d3js.org/d3.v3.js""></script>
    <script type="text/javascript" src="static/js/settings.js""></script>
    <style>
        body{ font: Arial 12px; text-align: center;}
    </style>
</head>

<body>
    <div class="banner">
        <h1>Face Grouper</h1>
    </div>
    <script type="text/javascript">
        dataset = {{data}};
        var svgElement = d3.select("body")
            .append("svg")
            .append("g"),
            
            nodes = dataset.nodes,
            links = dataset.links,
            force = d3.layout.force()
                .nodes(nodes)
                .links(links)
                .linkDistance(d => d.distance*distance_multiplier)
                .start(),
            link = svgElement.selectAll(".link")
                .data(links)
                .enter()
                .append("line")
                .attr("stroke-width", stroke_width)
                .attr("class", "link"),
            node = svgElement.selectAll(".node")
                .data(nodes)
                .enter()
                .append("g")
                .attr("class", "node")
                .call(force.drag);
            if(showText){
                node.append("text")
                        .attr("dx", 12)
                        .attr("dy", "0.35em")
                        //.attr("font-size", function (d) { return d.influence * 1.5 > 9 ? d.influence * 1.5 : 9; })
                        .text(function (d) { return d.name; });
            }
            if(showPictures){
                if(showText){
                    node.append("image")
                        .attr("xlink:href", function (d) { return picture_path + d.name  + picture_file_type; })
                        .attr("x", 10)
                        .attr("y", -50)
                        .attr("r", 10)
                        .attr("width", picture_width)
                        .attr("height", picture_height);
                } else {
                    node.append("image")
                        .attr("xlink:href", function (d) { return picture_path + d.name  + picture_file_type; })
                        .attr("x", -picture_width/2)
                        .attr("y", -picture_height/2)
                        .attr("r", 0)
                        .attr("width", picture_width)
                        .attr("height", picture_height);
                }
            }
        force.on("tick", function () {
            node
                .attr("cx", function (d) { return d.x; })
                .attr("cy", function (d) { return d.y; })
                .attr("transform", function (d) { return "translate(" + d.x + "," + d.y + ")"; });
            link
                .attr("x1", function (d) { return d.source.x; })
                .attr("y1", function (d) { return d.source.y; })
                .attr("x2", function (d) { return d.target.x; })
                .attr("y2", function (d) { return d.target.y; });
        });

        resize();
        d3.select(window).on("resize", resize);
        function resize() {
            width = window.innerWidth, height = window.innerHeight;
            svgElement.attr("width", width).attr("height", height);
            force.size([width, height]).resume();
        }
    </script>

</body>

</html>