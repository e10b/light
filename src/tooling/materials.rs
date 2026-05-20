use crate::material_editor::RuntimeMaterialPreview;
use crate::prism_file::{
    MaterialData as PrismMaterialData, NodeLink, NodeProperties, NodeType, ShaderNode,
};
use petgraph::visit::EdgeRef;

pub fn make_white_material() -> PrismMaterialData {
    PrismMaterialData {
        name: "White".to_string(),
        graph: {
            let mut g = petgraph::graph::DiGraph::new();
            let n_out = g.add_node(ShaderNode {
                node_type: NodeType::MaterialOutput,
                properties: NodeProperties::default(),
            });
            let n_in = g.add_node(ShaderNode {
                node_type: NodeType::FloatInput,
                properties: NodeProperties {
                    float_value: Some(1.0),
                    vec3_value: Some([1.0, 1.0, 1.0]),
                    roughness: None,
                    transmission: None,
                    ior: None,
                    bsdf_connected: None,
                },
            });
            g.add_edge(
                n_in,
                n_out,
                NodeLink {
                    output_socket: "Value".to_string(),
                    input_socket: "Surface".to_string(),
                },
            );
            g
        },
    }
}

pub fn make_empty_material() -> PrismMaterialData {
    PrismMaterialData {
        name: "Empty".to_string(),
        graph: {
            let mut g = petgraph::graph::DiGraph::new();
            g.add_node(ShaderNode {
                node_type: NodeType::MaterialOutput,
                properties: NodeProperties::default(),
            });
            g
        },
    }
}

pub fn make_glass_material() -> PrismMaterialData {
    PrismMaterialData {
        name: "Glass".to_string(),
        graph: {
            let mut g = petgraph::graph::DiGraph::new();
            let n_bsdf = g.add_node(ShaderNode {
                node_type: NodeType::PrincipledBSDF,
                properties: NodeProperties {
                    float_value: None,
                    vec3_value: Some([0.98, 1.0, 1.0]),
                    roughness: Some(0.02),
                    transmission: Some(1.0),
                    ior: Some(1.52),
                    bsdf_connected: Some(true),
                },
            });
            let n_out = g.add_node(ShaderNode {
                node_type: NodeType::MaterialOutput,
                properties: NodeProperties::default(),
            });
            g.add_edge(
                n_bsdf,
                n_out,
                NodeLink {
                    output_socket: "BSDF".to_string(),
                    input_socket: "Surface".to_string(),
                },
            );
            g
        },
    }
}

pub fn make_checker_material() -> PrismMaterialData {
    PrismMaterialData {
        name: "Checker".to_string(),
        graph: {
            let mut g = petgraph::graph::DiGraph::new();
            let n_checker = g.add_node(ShaderNode {
                node_type: NodeType::CheckerTexture,
                properties: NodeProperties {
                    float_value: Some(2.0),
                    vec3_value: Some([0.2, 0.2, 0.2]),
                    roughness: None,
                    transmission: None,
                    ior: None,
                    bsdf_connected: None,
                },
            });
            let n_bsdf = g.add_node(ShaderNode {
                node_type: NodeType::PrincipledBSDF,
                properties: NodeProperties {
                    float_value: None,
                    vec3_value: Some([0.5, 0.5, 0.5]),
                    roughness: Some(0.9),
                    transmission: Some(0.0),
                    ior: Some(1.0),
                    bsdf_connected: Some(true),
                },
            });
            let n_out = g.add_node(ShaderNode {
                node_type: NodeType::MaterialOutput,
                properties: NodeProperties::default(),
            });
            g.add_edge(
                n_checker,
                n_bsdf,
                NodeLink {
                    output_socket: "Color".to_string(),
                    input_socket: "Base Color".to_string(),
                },
            );
            g.add_edge(
                n_bsdf,
                n_out,
                NodeLink {
                    output_socket: "BSDF".to_string(),
                    input_socket: "Surface".to_string(),
                },
            );
            g
        },
    }
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
