use crate::blender_data::Id;

pub struct OutlinerItem {
    pub object_id: Id,
    pub label: String,
    pub selected: bool,
}

pub fn draw_outliner_panel(
    ctx: &egui::Context,
    items: &[OutlinerItem],
    project_status: &str,
) -> Option<Id> {
    let mut clicked = None;
    egui::SidePanel::left("outliner")
        .resizable(true)
        .default_width(230.0)
        .show(ctx, |ui| {
            ui.heading("Outliner");
            ui.label("Scene");
            ui.separator();
            for item in items {
                if ui
                    .selectable_label(item.selected, item.label.as_str())
                    .clicked()
                {
                    clicked = Some(item.object_id);
                }
            }
            if !project_status.is_empty() {
                ui.separator();
                ui.label(project_status);
            }
        });
    clicked
}
