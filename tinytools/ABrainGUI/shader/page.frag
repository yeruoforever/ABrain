# version 330 core

in vec2 tex_coord;
out vec4 color;

uniform sampler2D sense;

void main(){
     color=texture(sense, tex_coord);
}