#version 330 core

out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D debugTexture;

void main()
{
    vec3 color = texture(debugTexture, TexCoord).rgb;
    FragColor = vec4(color, 1.0);
}
