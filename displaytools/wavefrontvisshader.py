# Vertex shader
VS = """

uniform float z1;
uniform float z2;
uniform float z3;
uniform float z4;
uniform float z5;
uniform float z6;
uniform float min;
uniform float max;

float z1_mag(float x, float y) {return z1;}
float z2_mag(float x, float y) {return z2*2.0*x;}
float z3_mag(float x, float y) {return z3*2.0*y;}
float z4_mag(float x, float y) {return z4*sqrt(3.0)*(2.0*(x*x+y*y)-1.0);}
float z5_mag(float x, float y) {return z5*sqrt(6.0)*2.0*x*y;}
float z6_mag(float x, float y) {return z6*sqrt(6.0)*0.5*(x*x-y*y);}

float z1_dx(float x, float y) {return 0.0;}
float z2_dx(float x, float y) {return z2*2.0;}
float z3_dx(float x, float y) {return 0.0;}
float z4_dx(float x, float y) {return z4*sqrt(3.0)*(4.0*(x));}
float z5_dx(float x, float y) {return z5*sqrt(6.0)*2.0*y;}
float z6_dx(float x, float y) {return z6*sqrt(6.0)*(x);}

float z1_dy(float x, float y) {return 0.0;}
float z2_dy(float x, float y) {return 0.0;}
float z3_dy(float x, float y) {return z3*2.0;}
float z4_dy(float x, float y) {return z4*sqrt(3.0)*(4.0*(y));}
float z5_dy(float x, float y) {return z5*sqrt(6.0)*2.0*x;}
float z6_dy(float x, float y) {return z6*sqrt(6.0)*(y);}

varying vec4 color;
varying vec3 View;
varying vec3 Light;
varying vec3 Normal;

void main()
{
    //Compute the magnitude of the wavefront using the zernike functions
    float x = gl_Vertex.x;
    float z = gl_Vertex.z;
    float mag = z1_mag(x, z) + z2_mag(x, z) + z3_mag(x, z)
     + z4_mag(x, z) + z5_mag(x, z) + z6_mag(x, z);
    float dx = z1_dx(x, z) + z2_dx(x, z) + z3_dx(x, z)
     + z4_dx(x, z) + z5_dx(x, z) + z6_dx(x, z);
    float dy = z1_dy(x, z) + z2_dy(x, z) + z3_dy(x, z)
     + z4_dy(x, z) + z5_dy(x, z) + z6_dy(x, z);
     
    mag = (mag-min)/(max-min);
    
    //Set vertex color
    float redmag = (mag-0.0)/0.3;
    float greenmag = (mag-0.6)/0.3;
    float bluemag = (mag-0.9)/0.5;
    color = vec4(redmag, greenmag, bluemag, 1.0);
    
    //  Set vertex position
    gl_Position = gl_ModelViewProjectionMatrix * (gl_Vertex + 0.3*vec4(0.0, mag, 0.0, 1.0));
    
    vec3 P = vec3(gl_ModelViewMatrix * gl_Vertex);
    //  Light position
    Light  = vec3(gl_LightSource[0].position) - P;
    //  Normal from zernikie derivative functions    
    Normal = -gl_NormalMatrix*vec3(dx, -1.0, dy)/sqrt((dx*dx+1.0)*(dy*dy+1.0));
    
    //  Eye position
    View  = -P;
    
}
"""

# Fragment shader
FS = """
varying vec4 color;
varying vec3 View;
varying vec3 Light;
varying vec3 Normal;

uniform sampler2D tex;
uniform sampler2D normalMap;

vec4 phong()
{  
   vec3 N = normalize(Normal);
   //  L is the light vector
   vec3 L = normalize(Light);

   //  Emission and ambient color
   vec4 col = vec4(0.3, 0.3, 0.3, 1.0);

   //  Diffuse light is cosine of light and normal vectors
   float Id = dot(L,N);
   if (Id>0.0)
   {
      //  Add diffuse
      col += Id*vec4(0.3,0.3,0.3,1.0);
      //  R is the reflected light vector R = 2(L.N)N - L
      vec3 R = reflect(-L,N);
      //  V is the view vector (eye vector)
      vec3 V = normalize(View);
      //  Specular is cosine of reflected and view vectors
      float Is = dot(R,V);
      if (Is>0.0) col += pow(Is,20.0)*vec4(0.2, 0.2, 0.2, 1.0);
   }

   //  Return sum of color components
   return col; //vec4(L, 1.0);
}

void main()
{
   gl_FragColor = phong()*color;
}
"""