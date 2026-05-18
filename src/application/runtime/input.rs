use std::collections::HashSet;

use winit::event::{ElementState, KeyEvent, WindowEvent};

pub fn handle_pointer_window_event(
    event: &WindowEvent,
    mouse_pos: &mut [f32; 2],
    mouse_left_down: &mut bool,
    mouse_left_clicked: &mut bool,
    mouse_left_dragging: &mut bool,
) {
    match event {
        WindowEvent::CursorMoved { position, .. } => {
            *mouse_pos = [position.x as f32, position.y as f32];
            if *mouse_left_down {
                *mouse_left_dragging = true;
            }
        }
        WindowEvent::MouseInput {
            state,
            button: winit::event::MouseButton::Left,
            ..
        } => {
            *mouse_left_down = *state == ElementState::Pressed;
            if *state == ElementState::Pressed {
                *mouse_left_clicked = true;
                *mouse_left_dragging = false;
            }
        }
        _ => {}
    }
}

pub fn handle_keyboard_input(
    event: &KeyEvent,
    show_editor_ui: &mut bool,
    keys_pressed: &mut HashSet<String>,
) {
    match event.state {
        ElementState::Pressed => {
            if event.physical_key
                == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::F1)
            {
                if !event.repeat {
                    *show_editor_ui = !*show_editor_ui;
                }
            } else if let winit::keyboard::Key::Character(c) = &event.logical_key {
                keys_pressed.insert(c.to_lowercase().to_string());
            } else if event.physical_key
                == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Space)
            {
                keys_pressed.insert("Space".to_string());
            } else if event.physical_key
                == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ShiftLeft)
                || event.physical_key
                    == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ShiftRight)
            {
                keys_pressed.insert("Shift".to_string());
            } else if event.physical_key
                == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ControlLeft)
                || event.physical_key
                    == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ControlRight)
            {
                keys_pressed.insert("Control".to_string());
            } else if event.physical_key
                == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowUp)
            {
                keys_pressed.insert("ArrowUp".to_string());
            } else if event.physical_key
                == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowDown)
            {
                keys_pressed.insert("ArrowDown".to_string());
            } else if event.physical_key
                == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowLeft)
            {
                keys_pressed.insert("ArrowLeft".to_string());
            } else if event.physical_key
                == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowRight)
            {
                keys_pressed.insert("ArrowRight".to_string());
            }
        }
        ElementState::Released => {
            if let winit::keyboard::Key::Character(c) = &event.logical_key {
                keys_pressed.remove(&c.to_lowercase().to_string());
            } else if event.physical_key
                == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::Space)
            {
                keys_pressed.remove("Space");
            } else if event.physical_key
                == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ShiftLeft)
                || event.physical_key
                    == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ShiftRight)
            {
                keys_pressed.remove("Shift");
            } else if event.physical_key
                == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ControlLeft)
                || event.physical_key
                    == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ControlRight)
            {
                keys_pressed.remove("Control");
            } else if event.physical_key
                == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowUp)
            {
                keys_pressed.remove("ArrowUp");
            } else if event.physical_key
                == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowDown)
            {
                keys_pressed.remove("ArrowDown");
            } else if event.physical_key
                == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowLeft)
            {
                keys_pressed.remove("ArrowLeft");
            } else if event.physical_key
                == winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::ArrowRight)
            {
                keys_pressed.remove("ArrowRight");
            }
        }
    }
}
