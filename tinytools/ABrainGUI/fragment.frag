# version 330 core

in vec3 light_dir;

uniform float step;
uniform float alpha;
uniform float pix_min;
uniform float pix_max;
uniform vec3 eye;
uniform vec3 cube_a;
uniform vec3 cube_b;
uniform mat4 W2M;
uniform sampler3D volume;

out vec4 color;


const float EPSILON = 0.000001f;

float max3(float a, float b, float c){
    return max(a,max(b,c));
}

float min3(float a, float b, float c){
    return min(a,min(b,c));
}

vec3 maxv3(vec3 a,vec3 b){
    return vec3(max(a.x,b.x),max(a.y,b.y),max(a.z,b.z));
}

vec3 minv3(vec3 a, vec3 b){
    return vec3(min(a.x,b.x),min(a.y,b.y),min(a.z,b.z));
}

bool is_intersect(vec3 origin, vec3 light, vec3 cube_a, vec3 cube_b, out vec2 t){
    vec3 a = (cube_a-origin)/light;
    vec3 b = (cube_b-origin)/light;
    

    vec3 ta = minv3(a,b);
    vec3 tb = maxv3(a,b);

    float t_min = max3(ta.x,ta.y,ta.z);
    float t_max = min3(tb.x,tb.y,tb.z);

    t = vec2(t_min,t_max);

    if(t_min<0 || t_min>t_max)
        return false;
    else
        return true;
}

void main(){
    // color
    vec4 dest;
    vec4 src;
    // points
    vec3 pos;
    // texture coord
    vec3 tex_coord;
    vec3 tex_bound_0=vec3(0.0);
    vec3 tex_bound_1=vec3(1.0);
    // pixel value
    float pix_window;
    float pix_val;
    // light 
    vec3 light_step;
    vec2 t_range;

    dest = vec4(0);
    if(is_intersect(eye,light_dir,cube_a,cube_b,t_range)){
        pos = eye+light_dir*(EPSILON+t_range.x);
        light_step = light_dir*step;
        pix_window = pix_max-pix_min;
        for(int i=0;i<1000;i++){
            tex_coord = (W2M*vec4(pos,1.0)).xyz;
            if(any(lessThan(tex_coord,tex_bound_0))||any(greaterThan(tex_coord,tex_bound_1))||dest.a > 0.95)
                break;
            pix_val = texture(volume,tex_coord).r;
            pix_val = clamp(pix_val,pix_min,pix_max);
            pix_val = (pix_val-pix_min)/pix_window;
            src = vec4(pix_val);
            src.a *= alpha;
            src.rgb *= src.a;
            dest += (1-src.a)*src;
            if(dest.a > 0.99)
                break;
            pos = pos+light_step;
        }
        
    }
    color = dest;
}