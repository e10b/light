use crate::material_editor::RuntimeMaterialPreview;
use crate::prism_file::{
    MaterialData as PrismMaterialData, NodeLink, NodeProperties, NodeType, ShaderNode,
};
use crate::tooling::lua::material_scripts_dir;
use mlua::{Lua, Table, Value};
use petgraph::visit::EdgeRef;
use std::path::PathBuf;

fn parse_vec3(table: &Table, key: &str, fallback: [f32; 3]) -> [f32; 3] {
    let Ok(t) = table.get::<Table>(key) else {
        return fallback;
    };
    [
        t.get::<f32>(1).unwrap_or(fallback[0]),
        t.get::<f32>(2).unwrap_or(fallback[1]),
        t.get::<f32>(3).unwrap_or(fallback[2]),
    ]
}

fn parse_checker(table: &Table, preview: &mut RuntimeMaterialPreview) {
    let Ok(ct) = table.get::<Table>("checker") else {
        return;
    };
    preview.checker_enabled = ct.get::<bool>("enabled").unwrap_or(preview.checker_enabled);
    preview.checker_scale = ct
        .get::<f32>("scale")
        .map(|v| v.max(0.05))
        .unwrap_or(preview.checker_scale);
    preview.checker_color_a = parse_vec3(&ct, "color_a", preview.checker_color_a);
    preview.checker_color_b = parse_vec3(&ct, "color_b", preview.checker_color_b);
}

fn material_script_path(material_name: &str) -> PathBuf {
    material_scripts_dir().join(format!("{}.lua", material_name.to_ascii_lowercase()))
}

pub fn ensure_default_material_scripts() {
    let dir = material_scripts_dir();
    let _ = std::fs::create_dir_all(&dir);
    let defaults: [(&str, &str); 4] = [
        (
            "white.lua",
            r#"return {
  bsdf_connected = true,
  base_color = {1.0, 1.0, 1.0},
  roughness = 0.65,
  transmission = 0.0,
  ior = 1.0,
}"#,
        ),
        (
            "empty.lua",
            r#"return {
  bsdf_connected = false
}"#,
        ),
        (
            "glass.lua",
            r#"return {
  bsdf_connected = true,
  base_color = {0.98, 1.0, 1.0},
  roughness = 0.02,
  transmission = 1.0,
  ior = 1.52,
}"#,
        ),
        (
            "checker.lua",
            r#"return {
  bsdf_connected = true,
  base_color = {0.5, 0.5, 0.5},
  roughness = 0.9,
  transmission = 0.0,
  ior = 1.0,
  checker = {
    enabled = true,
    scale = 2.0,
    color_a = {0.2, 0.2, 0.2},
    color_b = {0.86, 0.86, 0.86},
  }
}"#,
        ),
    ];
    for (name, src) in defaults {
        let path = dir.join(name);
        if !path.exists() {
            let _ = std::fs::write(path, src);
        }
    }
}

pub fn preview_from_material_script(material_name: &str) -> Option<RuntimeMaterialPreview> {
    let path = material_script_path(material_name);
    let source = std::fs::read_to_string(path).ok()?;
    let lua = Lua::new();
    let value = lua.load(&source).eval::<Value>().ok()?;
    let table = match value {
        Value::Table(t) => t,
        _ => return None,
    };
    let mut preview = RuntimeMaterialPreview::default();
    preview.bsdf_connected = table
        .get::<bool>("bsdf_connected")
        .unwrap_or(preview.bsdf_connected);
    preview.base_color = parse_vec3(&table, "base_color", preview.base_color);
    preview.roughness = table
        .get::<f32>("roughness")
        .map(|v| v.clamp(0.001, 1.0))
        .unwrap_or(preview.roughness);
    preview.transmission = table
        .get::<f32>("transmission")
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(preview.transmission);
    preview.ior = table
        .get::<f32>("ior")
        .map(|v| v.max(1.0))
        .unwrap_or(preview.ior);
    parse_checker(&table, &mut preview);
    Some(preview)
}

pub fn material_from_preview(name: &str, preview: RuntimeMaterialPreview) -> PrismMaterialData {
    PrismMaterialData {
        name: name.to_string(),
        graph: {
            let mut g = petgraph::graph::DiGraph::new();
            let n_out = g.add_node(ShaderNode {
                node_type: NodeType::MaterialOutput,
                properties: NodeProperties::default(),
            });
            if preview.bsdf_connected {
                let n_bsdf = g.add_node(ShaderNode {
                    node_type: NodeType::PrincipledBSDF,
                    properties: NodeProperties {
                        float_value: None,
                        vec3_value: Some(preview.base_color),
                        roughness: Some(preview.roughness),
                        transmission: Some(preview.transmission),
                        ior: Some(preview.ior),
                        bsdf_connected: Some(true),
                    },
                });
                g.add_edge(
                    n_bsdf,
                    n_out,
                    NodeLink {
                        output_socket: "BSDF".to_string(),
                        input_socket: "Surface".to_string(),
                    },
                );
                if preview.checker_enabled {
                    let n_checker = g.add_node(ShaderNode {
                        node_type: NodeType::CheckerTexture,
                        properties: NodeProperties {
                            float_value: Some(preview.checker_scale.max(0.05)),
                            vec3_value: Some(preview.checker_color_a),
                            roughness: None,
                            transmission: None,
                            ior: None,
                            bsdf_connected: None,
                        },
                    });
                    g.add_edge(
                        n_checker,
                        n_bsdf,
                        NodeLink {
                            output_socket: "Color".to_string(),
                            input_socket: "Base Color".to_string(),
                        },
                    );
                }
            }
            g
        },
    }
}

pub fn scripted_preview_or_graph(
    material_name: &str,
    material: Option<&PrismMaterialData>,
) -> RuntimeMaterialPreview {
    preview_from_material_script(material_name).unwrap_or_else(|| preview_from_material_data(material))
}

pub fn material_script_source_from_preview(
    _material_name: &str,
    preview: RuntimeMaterialPreview,
) -> String {
    let mut out = String::new();
    out.push_str("return {\n");
    out.push_str(&format!(
        "  bsdf_connected = {},\n",
        if preview.bsdf_connected { "true" } else { "false" }
    ));
    out.push_str(&format!(
        "  base_color = {{{:.6}, {:.6}, {:.6}}},\n",
        preview.base_color[0], preview.base_color[1], preview.base_color[2]
    ));
    out.push_str(&format!("  roughness = {:.6},\n", preview.roughness));
    out.push_str(&format!("  transmission = {:.6},\n", preview.transmission));
    out.push_str(&format!("  ior = {:.6},\n", preview.ior));
    if preview.checker_enabled {
        out.push_str("  checker = {\n");
        out.push_str("    enabled = true,\n");
        out.push_str(&format!("    scale = {:.6},\n", preview.checker_scale));
        out.push_str(&format!(
            "    color_a = {{{:.6}, {:.6}, {:.6}}},\n",
            preview.checker_color_a[0], preview.checker_color_a[1], preview.checker_color_a[2]
        ));
        out.push_str(&format!(
            "    color_b = {{{:.6}, {:.6}, {:.6}}},\n",
            preview.checker_color_b[0], preview.checker_color_b[1], preview.checker_color_b[2]
        ));
        out.push_str("  },\n");
    }
    out.push_str("}\n");
    out
}

pub fn make_white_material() -> PrismMaterialData {
    let preview = preview_from_material_script("White").unwrap_or_else(|| RuntimeMaterialPreview {
        bsdf_connected: true,
        base_color: [1.0, 1.0, 1.0],
        roughness: 0.65,
        transmission: 0.0,
        ior: 1.0,
        ..RuntimeMaterialPreview::default()
    });
    material_from_preview("White", preview)
}

pub fn make_empty_material() -> PrismMaterialData {
    let preview = preview_from_material_script("Empty").unwrap_or_else(|| RuntimeMaterialPreview {
        bsdf_connected: false,
        ..RuntimeMaterialPreview::default()
    });
    material_from_preview("Empty", preview)
}

pub fn make_glass_material() -> PrismMaterialData {
    let preview = preview_from_material_script("Glass").unwrap_or_else(|| RuntimeMaterialPreview {
        bsdf_connected: true,
        base_color: [0.98, 1.0, 1.0],
        roughness: 0.02,
        transmission: 1.0,
        ior: 1.52,
        ..RuntimeMaterialPreview::default()
    });
    material_from_preview("Glass", preview)
}

pub fn make_checker_material() -> PrismMaterialData {
    let preview = preview_from_material_script("Checker").unwrap_or_else(|| RuntimeMaterialPreview {
        bsdf_connected: true,
        base_color: [0.5, 0.5, 0.5],
        roughness: 0.9,
        transmission: 0.0,
        ior: 1.0,
        checker_enabled: true,
        checker_scale: 2.0,
        checker_color_a: [0.2, 0.2, 0.2],
        checker_color_b: [0.86, 0.86, 0.86],
    });
    material_from_preview("Checker", preview)
}

pub fn preview_from_material_data(material: Option<&PrismMaterialData>) -> RuntimeMaterialPreview {
    let mut out = RuntimeMaterialPreview::default();
    let Some(material) = material else {
        return out;
    };
    let mut bsdf_idx = None;
    let mut out_idx = None;
    for idx in material.graph.node_indices() {
        match material.graph[idx].node_type {
            NodeType::PrincipledBSDF => bsdf_idx = Some(idx),
            NodeType::MaterialOutput => out_idx = Some(idx),
            _ => {}
        }
    }
    if let Some(bi) = bsdf_idx {
        let props = &material.graph[bi].properties;
        if let Some(v) = props.vec3_value {
            out.base_color = v;
        }
        if let Some(v) = props.roughness {
            out.roughness = v;
        }
        if let Some(v) = props.transmission {
            out.transmission = v;
        }
        if let Some(v) = props.ior {
            out.ior = v;
        }
    }
    if let Some(bi) = bsdf_idx {
        out.checker_enabled = material.graph.edges_directed(bi, petgraph::Direction::Incoming).any(|edge| {
            let src = edge.source();
            edge.weight().input_socket == "Base Color"
                && edge.weight().output_socket == "Color"
                && matches!(material.graph[src].node_type, NodeType::CheckerTexture)
        });
        if out.checker_enabled {
            for edge in material.graph.edges_directed(bi, petgraph::Direction::Incoming) {
                let src = edge.source();
                if edge.weight().input_socket == "Base Color"
                    && edge.weight().output_socket == "Color"
                    && matches!(material.graph[src].node_type, NodeType::CheckerTexture)
                {
                    if let Some(v) = material.graph[src].properties.float_value {
                        out.checker_scale = v.max(0.05);
                    }
                    if let Some(v) = material.graph[src].properties.vec3_value {
                        out.checker_color_a = v;
                    }
                    break;
                }
            }
        }
    }
    if let (Some(bi), Some(oi)) = (bsdf_idx, out_idx) {
        out.bsdf_connected = material.graph.edges_connecting(bi, oi).any(|edge| {
            edge.weight().output_socket == "BSDF" && edge.weight().input_socket == "Surface"
        });
        if let Some(v) = material.graph[bi].properties.bsdf_connected {
            out.bsdf_connected = v;
        }
    }
    out.roughness = out.roughness.clamp(0.001, 1.0);
    out.transmission = out.transmission.clamp(0.0, 1.0);
    out.ior = out.ior.max(1.0);
    out
}
