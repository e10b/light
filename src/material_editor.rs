use std::borrow::Cow;
use std::collections::HashMap;

use egui_node_editor::{
    DataTypeTrait, Graph, GraphEditorState, InputId, InputParamKind, NodeDataTrait, NodeId,
    NodeResponse, NodeTemplateIter, NodeTemplateTrait, OutputId, UserResponseTrait,
    WidgetValueTrait,
};
use petgraph::visit::EdgeRef;

use crate::prism_file::{MaterialData, NodeType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SocketType {
    Scalar,
    Vector3,
    Shader,
}

impl DataTypeTrait<()> for SocketType {
    fn data_type_color(&self, _user_state: &mut ()) -> egui::Color32 {
        match self {
            SocketType::Scalar => egui::Color32::from_rgb(214, 169, 76),
            SocketType::Vector3 => egui::Color32::from_rgb(112, 188, 255),
            SocketType::Shader => egui::Color32::from_rgb(198, 130, 255),
        }
    }

    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(match self {
            SocketType::Scalar => "Scalar",
            SocketType::Vector3 => "Vector3",
            SocketType::Shader => "Shader",
        })
    }
}

#[derive(Clone, Default, Debug)]
enum SocketValue {
    Scalar(f32),
    Vector3([f32; 3]),
    #[default]
    None,
}

#[derive(Clone, Debug)]
struct GraphNodeData {
    node_type: NodeType,
}

#[derive(Clone, Debug)]
enum GraphResponse {}
impl UserResponseTrait for GraphResponse {}

impl WidgetValueTrait for SocketValue {
    type Response = GraphResponse;
    type UserState = ();
    type NodeData = GraphNodeData;

    fn value_widget(
        &mut self,
        param_name: &str,
        _node_id: NodeId,
        ui: &mut egui::Ui,
        _user_state: &mut Self::UserState,
        _node_data: &Self::NodeData,
    ) -> Vec<Self::Response> {
        match self {
            SocketValue::Scalar(v) => {
                ui.horizontal(|ui| {
                    ui.label(param_name);
                    ui.add(egui::DragValue::new(v).speed(0.01));
                });
            }
            SocketValue::Vector3(v) => {
                ui.label(param_name);
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut v[0]).speed(0.01));
                    ui.add(egui::DragValue::new(&mut v[1]).speed(0.01));
                    ui.add(egui::DragValue::new(&mut v[2]).speed(0.01));
                });
            }
            SocketValue::None => {
                ui.label(param_name);
            }
        }
        Vec::new()
    }
}

impl NodeDataTrait for GraphNodeData {
    type Response = GraphResponse;
    type UserState = ();
    type DataType = SocketType;
    type ValueType = SocketValue;

    fn bottom_ui(
        &self,
        ui: &mut egui::Ui,
        _node_id: NodeId,
        _graph: &Graph<Self, Self::DataType, Self::ValueType>,
        _user_state: &mut Self::UserState,
    ) -> Vec<NodeResponse<Self::Response, Self>> {
        ui.small(format!("{:?}", self.node_type));
        Vec::new()
    }
}

#[derive(Clone)]
struct DummyNodeTemplate;

impl NodeTemplateTrait for DummyNodeTemplate {
    type NodeData = GraphNodeData;
    type DataType = SocketType;
    type ValueType = SocketValue;
    type UserState = ();
    type CategoryType = ();

    fn node_finder_label(&self, _user_state: &mut Self::UserState) -> Cow<'_, str> {
        Cow::Borrowed("Node")
    }

    fn node_graph_label(&self, _user_state: &mut Self::UserState) -> String {
        "Node".to_string()
    }

    fn user_data(&self, _user_state: &mut Self::UserState) -> Self::NodeData {
        GraphNodeData {
            node_type: NodeType::MaterialOutput,
        }
    }

    fn build_node(
        &self,
        _graph: &mut Graph<Self::NodeData, Self::DataType, Self::ValueType>,
        _user_state: &mut Self::UserState,
        _node_id: NodeId,
    ) {
    }
}

struct EmptyTemplates;
impl NodeTemplateIter for EmptyTemplates {
    type Item = DummyNodeTemplate;
    fn all_kinds(&self) -> Vec<Self::Item> {
        Vec::new()
    }
}

pub struct MaterialGraphEditor {
    state: GraphEditorState<GraphNodeData, SocketType, SocketValue, DummyNodeTemplate, ()>,
    loaded_key: Option<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct RuntimeMaterialPreview {
    pub bsdf_connected: bool,
    pub checker_enabled: bool,
    pub checker_scale: f32,
    pub checker_color_a: [f32; 3],
    pub checker_color_b: [f32; 3],
    pub base_color: [f32; 3],
    pub roughness: f32,
    pub transmission: f32,
    pub ior: f32,
}

impl Default for RuntimeMaterialPreview {
    fn default() -> Self {
        Self {
            bsdf_connected: false,
            checker_enabled: false,
            checker_scale: 2.0,
            checker_color_a: [0.2, 0.2, 0.2],
            checker_color_b: [0.86, 0.86, 0.86],
            base_color: [1.0, 1.0, 1.0],
            roughness: 0.65,
            transmission: 0.0,
            ior: 1.0,
        }
    }
}

impl MaterialGraphEditor {
    pub fn new() -> Self {
        Self {
            state: GraphEditorState::new(1.0),
            loaded_key: None,
        }
    }

    pub fn load_material(&mut self, key: &str, material: &MaterialData) {
        if self.loaded_key.as_deref() == Some(key) {
            return;
        }

        self.state = GraphEditorState::new(1.0);
        self.loaded_key = Some(key.to_string());

        let mut node_map = HashMap::new();
        let mut input_sockets: HashMap<(NodeId, String), InputId> = HashMap::new();
        let mut output_sockets: HashMap<(NodeId, String), OutputId> = HashMap::new();

        for (idx, node) in material.graph.node_indices().enumerate() {
            let shader_node = &material.graph[node];
            let label = format!("{:?}", shader_node.node_type);
            let node_id = self.state.graph.add_node(
                label,
                GraphNodeData {
                    node_type: shader_node.node_type.clone(),
                },
                |graph, nid| match shader_node.node_type {
                    NodeType::FloatInput => {
                        let value = shader_node.properties.float_value.unwrap_or(0.0);
                        let out =
                            graph.add_output_param(nid, "Value".to_string(), SocketType::Scalar);
                        output_sockets.insert((nid, "Value".to_string()), out);
                        let _ = graph.add_input_param(
                            nid,
                            "Value".to_string(),
                            SocketType::Scalar,
                            SocketValue::Scalar(value),
                            InputParamKind::ConstantOnly,
                            true,
                        );
                    }
                    NodeType::VectorMath => {
                        let a = graph.add_input_param(
                            nid,
                            "A".to_string(),
                            SocketType::Scalar,
                            SocketValue::Scalar(0.0),
                            InputParamKind::ConnectionOrConstant,
                            true,
                        );
                        input_sockets.insert((nid, "A".to_string()), a);
                        let b = graph.add_input_param(
                            nid,
                            "B".to_string(),
                            SocketType::Scalar,
                            SocketValue::Scalar(0.0),
                            InputParamKind::ConnectionOrConstant,
                            true,
                        );
                        input_sockets.insert((nid, "B".to_string()), b);
                        let out =
                            graph.add_output_param(nid, "Value".to_string(), SocketType::Scalar);
                        output_sockets.insert((nid, "Value".to_string()), out);
                    }
                    NodeType::CheckerTexture => {
                        let color_a = graph.add_input_param(
                            nid,
                            "Color A".to_string(),
                            SocketType::Vector3,
                            SocketValue::Vector3(
                                shader_node.properties.vec3_value.unwrap_or([0.2, 0.2, 0.2]),
                            ),
                            InputParamKind::ConnectionOrConstant,
                            true,
                        );
                        input_sockets.insert((nid, "Color A".to_string()), color_a);
                        let color_b = graph.add_input_param(
                            nid,
                            "Color B".to_string(),
                            SocketType::Vector3,
                            SocketValue::Vector3([0.86, 0.86, 0.86]),
                            InputParamKind::ConnectionOrConstant,
                            true,
                        );
                        input_sockets.insert((nid, "Color B".to_string()), color_b);
                        let scale = graph.add_input_param(
                            nid,
                            "Scale".to_string(),
                            SocketType::Scalar,
                            SocketValue::Scalar(shader_node.properties.float_value.unwrap_or(2.0)),
                            InputParamKind::ConnectionOrConstant,
                            true,
                        );
                        input_sockets.insert((nid, "Scale".to_string()), scale);
                        let out =
                            graph.add_output_param(nid, "Color".to_string(), SocketType::Vector3);
                        output_sockets.insert((nid, "Color".to_string()), out);
                    }
                    NodeType::PrincipledBSDF => {
                        let base = graph.add_input_param(
                            nid,
                            "Base Color".to_string(),
                            SocketType::Vector3,
                            SocketValue::Vector3(
                                shader_node.properties.vec3_value.unwrap_or([1.0, 1.0, 1.0]),
                            ),
                            InputParamKind::ConnectionOrConstant,
                            true,
                        );
                        input_sockets.insert((nid, "Base Color".to_string()), base);
                        let rough = graph.add_input_param(
                            nid,
                            "Roughness".to_string(),
                            SocketType::Scalar,
                            SocketValue::Scalar(shader_node.properties.roughness.unwrap_or(0.03)),
                            InputParamKind::ConnectionOrConstant,
                            true,
                        );
                        input_sockets.insert((nid, "Roughness".to_string()), rough);
                        let trans = graph.add_input_param(
                            nid,
                            "Transmission".to_string(),
                            SocketType::Scalar,
                            SocketValue::Scalar(shader_node.properties.transmission.unwrap_or(1.0)),
                            InputParamKind::ConnectionOrConstant,
                            true,
                        );
                        input_sockets.insert((nid, "Transmission".to_string()), trans);
                        let ior = graph.add_input_param(
                            nid,
                            "IOR".to_string(),
                            SocketType::Scalar,
                            SocketValue::Scalar(shader_node.properties.ior.unwrap_or(1.52)),
                            InputParamKind::ConnectionOrConstant,
                            true,
                        );
                        input_sockets.insert((nid, "IOR".to_string()), ior);
                        let out =
                            graph.add_output_param(nid, "BSDF".to_string(), SocketType::Shader);
                        output_sockets.insert((nid, "BSDF".to_string()), out);
                    }
                    NodeType::MaterialOutput => {
                        let surf = graph.add_input_param(
                            nid,
                            "Surface".to_string(),
                            SocketType::Shader,
                            SocketValue::None,
                            InputParamKind::ConnectionOnly,
                            true,
                        );
                        input_sockets.insert((nid, "Surface".to_string()), surf);
                    }
                },
            );
            node_map.insert(node, node_id);
            self.state.node_order.push(node_id);
            self.state
                .node_positions
                .insert(node_id, egui::pos2(32.0 + idx as f32 * 220.0, 64.0));
        }

        for edge in material.graph.edge_references() {
            let src = node_map[&edge.source()];
            let dst = node_map[&edge.target()];
            let link = edge.weight();
            if let (Some(out_id), Some(in_id)) = (
                output_sockets.get(&(src, link.output_socket.clone())),
                input_sockets.get(&(dst, link.input_socket.clone())),
            ) {
                self.state.graph.add_connection(*out_id, *in_id);
            }
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        let mut user_state = ();
        let _ = self
            .state
            .draw_graph_editor(ui, EmptyTemplates, &mut user_state, Vec::new());
    }

    pub fn runtime_preview(&self) -> RuntimeMaterialPreview {
        let mut preview = RuntimeMaterialPreview::default();
        let mut bsdf_nodes = Vec::new();
        let mut material_output_surface_input = None;

        for (node_id, node) in self.state.graph.nodes.iter() {
            match node.user_data.node_type {
                NodeType::PrincipledBSDF => bsdf_nodes.push(node_id),
                NodeType::MaterialOutput => {
                    if let Ok(surface_id) = node.get_input("Surface") {
                        material_output_surface_input = Some(surface_id);
                    }
                }
                _ => {}
            }
        }

        let Some(surface_input) = material_output_surface_input else {
            return preview;
        };
        let Some(surface_src) = self.state.graph.connection(surface_input) else {
            return preview;
        };
        let source_node = self.state.graph.get_output(surface_src).node;
        if !bsdf_nodes.contains(&source_node) {
            return preview;
        }
        preview.bsdf_connected = true;

        let Some(node) = self.state.graph.nodes.get(source_node) else {
            return preview;
        };
        for (name, input_id) in &node.inputs {
            let Some(inp) = self.state.graph.inputs.get(*input_id) else {
                continue;
            };
            match (name.as_str(), &inp.value) {
                ("Base Color", SocketValue::Vector3(v)) => preview.base_color = *v,
                ("Roughness", SocketValue::Scalar(v)) => preview.roughness = *v,
                ("Transmission", SocketValue::Scalar(v)) => preview.transmission = *v,
                ("IOR", SocketValue::Scalar(v)) => preview.ior = *v,
                _ => {}
            }
        }
        if let Ok(base_color_input) = node.get_input("Base Color") {
            if let Some(base_src) = self.state.graph.connection(base_color_input) {
                let base_source_node = self.state.graph.get_output(base_src).node;
                if let Some(src_node) = self.state.graph.nodes.get(base_source_node) {
                    preview.checker_enabled = matches!(src_node.user_data.node_type, NodeType::CheckerTexture);
                    if preview.checker_enabled {
                        if let Ok(color_a_input) = src_node.get_input("Color A") {
                            if let Some(color_a_param) = self.state.graph.inputs.get(color_a_input) {
                                if let SocketValue::Vector3(v) = color_a_param.value {
                                    preview.checker_color_a = v;
                                }
                            }
                        }
                        if let Ok(color_b_input) = src_node.get_input("Color B") {
                            if let Some(color_b_param) = self.state.graph.inputs.get(color_b_input) {
                                if let SocketValue::Vector3(v) = color_b_param.value {
                                    preview.checker_color_b = v;
                                }
                            }
                        }
                        if let Ok(scale_input) = src_node.get_input("Scale") {
                            if let Some(scale_param) = self.state.graph.inputs.get(scale_input) {
                                if let SocketValue::Scalar(v) = scale_param.value {
                                    preview.checker_scale = v.max(0.05);
                                }
                            }
                        }
                    }
                }
            }
        }
        preview.roughness = preview.roughness.clamp(0.001, 1.0);
        preview.transmission = preview.transmission.clamp(0.0, 1.0);
        preview.ior = preview.ior.max(1.0);
        preview
    }

    pub fn commit_to_material(&self, material: &mut MaterialData) {
        let preview = self.runtime_preview();
        let mut bsdf_node_idx = None;
        let mut output_node_idx = None;
        for idx in material.graph.node_indices() {
            match material.graph[idx].node_type {
                NodeType::PrincipledBSDF => bsdf_node_idx = Some(idx),
                NodeType::MaterialOutput => output_node_idx = Some(idx),
                _ => {}
            }
        }
        if let Some(bsdf_idx) = bsdf_node_idx {
            let props = &mut material.graph[bsdf_idx].properties;
            props.vec3_value = Some(preview.base_color);
            props.roughness = Some(preview.roughness);
            props.transmission = Some(preview.transmission);
            props.ior = Some(preview.ior);
            props.bsdf_connected = Some(preview.bsdf_connected);
        }
        if let (Some(bsdf_idx), Some(output_idx)) = (bsdf_node_idx, output_node_idx) {
            let existing: Vec<_> = material
                .graph
                .edges_connecting(bsdf_idx, output_idx)
                .map(|e| e.id())
                .collect();
            for edge_id in existing {
                material.graph.remove_edge(edge_id);
            }
            if preview.bsdf_connected {
                material.graph.add_edge(
                    bsdf_idx,
                    output_idx,
                    crate::prism_file::NodeLink {
                        output_socket: "BSDF".to_string(),
                        input_socket: "Surface".to_string(),
                    },
                );
            }
        }
    }
}
