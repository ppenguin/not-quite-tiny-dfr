use crate::style::Color;
use libc::c_void;
use serde::Deserialize;
use std::{
    collections::HashMap,
    io::Read,
    os::fd::{AsRawFd, OwnedFd, RawFd},
    process::{Command, Stdio},
    sync::{Arc, Condvar, Mutex, MutexGuard},
    thread,
    time::{Duration, Instant},
};

/// Hard cap so a hung script can't wedge its worker thread forever.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(30);
/// First retry delay after a widget command fails (non-zero exit); doubles per
/// consecutive failure, capped at the widget's interval. Keeps a fetch that
/// failed because e.g. the network wasn't up yet from squatting on stale
/// output for a whole long interval.
const RETRY_BASE: Duration = Duration::from_secs(15);
/// Floor on the poll interval so nobody cooks their CPU with `Interval = 0.01`.
const MIN_INTERVAL: Duration = Duration::from_millis(100);
/// While a widget script runs, how often its worker re-checks the timeout and
/// the stop flag. Output is read as it arrives, so this only bounds how long a
/// stopped runtime's script can outlive it.
const CHILD_CHECK: Duration = Duration::from_secs(1);
/// How often the idle slider setter re-checks the stop flag; new values wake it
/// at once.
const SETTER_CHECK: Duration = Duration::from_secs(1);

/// A flag that sleeping threads can wait on: `wait` returns once the flag is
/// set or the timeout passes, instead of polling. `take` consumes the flag for
/// one-shot wakeups; the runtime's stop flag stays set.
#[derive(Default)]
struct Signal {
    flag: Mutex<bool>,
    cv: Condvar,
}

impl Signal {
    fn set(&self) {
        *self.flag.lock().unwrap() = true;
        self.cv.notify_all();
    }

    fn is_set(&self) -> bool {
        *self.flag.lock().unwrap()
    }

    /// Wait up to `timeout` for the flag; returns whether it is set.
    fn wait(&self, timeout: Duration) -> bool {
        let flag = self.flag.lock().unwrap();
        let (flag, _) = self.cv.wait_timeout_while(flag, timeout, |f| !*f).unwrap();
        *flag
    }

    /// Like `wait`, but clears the flag on the way out.
    fn take(&self, timeout: Duration) {
        let flag = self.flag.lock().unwrap();
        let (mut flag, _) = self.cv.wait_timeout_while(flag, timeout, |f| !*f).unwrap();
        *flag = false;
    }
}

/// Each widget worker's refresh signal, by widget id, so a tap can re-run a
/// command widget instead of waiting out its interval. Replaced by every new
/// runtime; a stale entry can only poke a thread that is already stopping.
static REFRESH: Mutex<Vec<(usize, Arc<Signal>)>> = Mutex::new(Vec::new());

/// Run a command widget's OnClick command off the render path, then re-run
/// the widget's own command so its output follows the tap at once.
pub fn run_tap_command(command: &str, id: usize) {
    let command = command.to_string();
    thread::spawn(move || {
        if let Err(e) = Command::new("sh").arg("-c").arg(&command).status() {
            eprintln!("not-quite-tiny-dfr: failed to run OnClick command {command:?}: {e}");
        }
        let refresh = REFRESH
            .lock()
            .unwrap()
            .iter()
            .find(|(i, _)| *i == id)
            .map(|(_, s)| s.clone());
        if let Some(refresh) = refresh {
            refresh.set();
        }
    });
}

/// How long after a slider set to distrust polled get results: a poll that
/// STARTED before the set can finish after it carrying the pre-set value,
/// and applying it would snap the slider's fill backwards.
const SET_QUIET: Duration = Duration::from_millis(1500);

/// A command widget to run: unique id, the shell command, and how often to run.
#[derive(Clone)]
pub struct WidgetSpec {
    pub id: usize,
    pub command: String,
    pub interval: Duration,
}

/// A slider widget: `get_command` prints the current value (0-100, optionally
/// followed by the word "muted") and is polled like a command widget;
/// `set_command` is run with `{}` replaced by the new value whenever the user
/// moves the slider; `mute_command` (optional) is run with `{}` replaced by
/// "toggle" (icon tap) or "0" (a drag unmutes).
#[derive(Clone)]
pub struct SliderSpec {
    pub id: usize,
    pub get_command: String,
    pub set_command: String,
    pub mute_command: Option<String>,
    pub interval: Duration,
}

/// Substitute `arg` into a command template: replaces `{}` when present,
/// otherwise appends it as an argument.
fn fill_placeholder(command: &str, arg: &str) -> String {
    if command.contains("{}") {
        command.replace("{}", arg)
    } else {
        format!("{command} {arg}")
    }
}

/// All the command-running widgets a config produced.
#[derive(Default)]
pub struct Widgets {
    pub commands: Vec<WidgetSpec>,
    pub sliders: Vec<SliderSpec>,
}

impl WidgetSpec {
    /// Turn a configured `Interval` (seconds) into a clamped `Duration`.
    pub fn interval_from_secs(secs: Option<f64>) -> Duration {
        let secs = secs.unwrap_or(2.0);
        if secs.is_finite() && secs > 0.0 {
            Duration::from_secs_f64(secs).max(MIN_INTERVAL)
        } else {
            MIN_INTERVAL
        }
    }
}

/// The latest output of a widget script.
#[derive(Clone, Default, PartialEq)]
pub struct WidgetOutput {
    pub text: String,
    pub color: Option<Color>,
    pub icon: Option<String>,
}

/// The optional JSON form a script may print for richer control.
#[derive(Deserialize)]
struct WidgetJson {
    text: Option<String>,
    color: Option<String>,
    icon: Option<String>,
}

/// Owns the worker threads that poll widget commands. Dropping it signals the
/// threads to stop; they are detached (not joined) so a config reload stays
/// snappy even if a script is mid-run -- the old threads notice the flag and
/// exit on their own.
pub struct WidgetRuntime {
    results: Arc<Mutex<HashMap<usize, WidgetOutput>>>,
    stop: Arc<Signal>,
    /// Wakes the setter thread when a set or mute is queued.
    setter_wake: Arc<Signal>,
    /// Latest slider value awaiting its set command, per slider id. The setter
    /// thread drains this; a fast drag coalesces to the newest value instead
    /// of queueing a process run per pixel.
    pending_sets: Arc<Mutex<HashMap<usize, i32>>>,
    set_commands: HashMap<usize, String>,
    /// Fully-substituted mute commands awaiting the setter thread.
    pending_mutes: Arc<Mutex<HashMap<usize, String>>>,
    mute_commands: HashMap<usize, String>,
    /// When each slider was last set or (un)muted, for the SET_QUIET window.
    last_set: Arc<Mutex<HashMap<usize, Instant>>>,
    /// Every worker's refresh signal, set on drop so sleeping workers see the
    /// stop flag at once.
    refresh: Vec<Arc<Signal>>,
}

impl WidgetRuntime {
    /// Spawn a worker thread per widget. `wake` is the write end of a pipe whose
    /// read end lives in the main epoll loop; a byte is written whenever a
    /// widget's output changes, so the loop wakes and redraws.
    pub fn new(widgets: Widgets, wake: Arc<OwnedFd>) -> WidgetRuntime {
        let results: Arc<Mutex<HashMap<usize, WidgetOutput>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let stop = Arc::new(Signal::default());
        let setter_wake = Arc::new(Signal::default());
        let mut refresh_by_id = Vec::new();
        for spec in widgets.commands {
            let results = results.clone();
            let stop = stop.clone();
            let wake = wake.clone();
            let refresh = Arc::new(Signal::default());
            refresh_by_id.push((spec.id, refresh.clone()));
            thread::spawn(move || run_widget(spec, results, stop, refresh, wake));
        }
        // Sliders poll their get command through the same worker path.
        let mut set_commands = HashMap::new();
        for spec in &widgets.sliders {
            set_commands.insert(spec.id, spec.set_command.clone());
            let poll = WidgetSpec {
                id: spec.id,
                command: spec.get_command.clone(),
                interval: spec.interval,
            };
            let results = results.clone();
            let stop = stop.clone();
            let wake = wake.clone();
            let refresh = Arc::new(Signal::default());
            refresh_by_id.push((poll.id, refresh.clone()));
            thread::spawn(move || run_widget(poll, results, stop, refresh, wake));
        }
        let refresh = refresh_by_id.iter().map(|(_, s)| s.clone()).collect();
        *REFRESH.lock().unwrap() = refresh_by_id;
        let pending_sets: Arc<Mutex<HashMap<usize, i32>>> = Arc::new(Mutex::new(HashMap::new()));
        let pending_mutes: Arc<Mutex<HashMap<usize, String>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let mute_commands: HashMap<usize, String> = widgets
            .sliders
            .iter()
            .filter_map(|s| Some((s.id, s.mute_command.clone()?)))
            .collect();
        if !widgets.sliders.is_empty() {
            let pending = pending_sets.clone();
            let mutes = pending_mutes.clone();
            let stop_flag = stop.clone();
            let commands = set_commands.clone();
            let get_commands: HashMap<usize, String> = widgets
                .sliders
                .iter()
                .map(|s| (s.id, s.get_command.clone()))
                .collect();
            let setter_results = results.clone();
            let wake = setter_wake.clone();
            thread::spawn(move || {
                run_setter(pending, mutes, commands, get_commands, setter_results, stop_flag, wake)
            });
        }
        WidgetRuntime {
            results,
            stop,
            setter_wake,
            pending_sets,
            set_commands,
            pending_mutes,
            mute_commands,
            last_set: Arc::new(Mutex::new(HashMap::new())),
            refresh,
        }
    }

    pub fn results(&self) -> MutexGuard<'_, HashMap<usize, WidgetOutput>> {
        self.results.lock().unwrap()
    }

    /// Queue a slider's set command with the new value. Coalesces: only the
    /// latest value per slider is kept, and the setter thread runs the actual
    /// command off the render path.
    pub fn set_slider(&self, id: usize, value: i32) {
        if !self.set_commands.contains_key(&id) {
            return;
        }
        self.last_set.lock().unwrap().insert(id, Instant::now());
        // Reflect the value in the results cache right away, so a post-drag
        // apply can't snap the fill back to a pre-drag poll reading.
        self.results.lock().unwrap().insert(
            id,
            WidgetOutput {
                text: value.to_string(),
                color: None,
                icon: None,
            },
        );
        self.pending_sets.lock().unwrap().insert(id, value);
        self.setter_wake.set();
    }

    /// Queue a slider's mute command. `arg` is "toggle" (icon tap) or "0"
    /// (deterministic unmute when a drag moves the volume).
    pub fn set_slider_mute(&self, id: usize, arg: &str) {
        if let Some(command) = self.mute_commands.get(&id) {
            self.last_set.lock().unwrap().insert(id, Instant::now());
            self.pending_mutes
                .lock()
                .unwrap()
                .insert(id, fill_placeholder(command, arg));
            self.setter_wake.set();
        }
    }

    /// Whether `id` was set recently enough that polled get results may still
    /// predate the set and should not be applied.
    pub fn recently_set(&self, id: usize) -> bool {
        self.last_set
            .lock()
            .unwrap()
            .get(&id)
            .is_some_and(|t| t.elapsed() < SET_QUIET)
    }
}

/// Drain queued slider set commands, running each to completion so no zombie
/// processes accumulate. One thread serves all sliders: set commands are
/// near-instant (wpctl, brightnessctl), and coalescing means a laggy one only
/// skips intermediate values, never queues them up.
fn run_setter(
    pending: Arc<Mutex<HashMap<usize, i32>>>,
    pending_mutes: Arc<Mutex<HashMap<usize, String>>>,
    commands: HashMap<usize, String>,
    get_commands: HashMap<usize, String>,
    results: Arc<Mutex<HashMap<usize, WidgetOutput>>>,
    stop: Arc<Signal>,
    wake: Arc<Signal>,
) {
    // Refresh the cache from the get command once a set/mute has been
    // applied, so the poller's next read agrees -- unless the drag already
    // queued a newer value, in which case that one is about to run anyway.
    let refresh = |id: usize, stop: &Signal| {
        if let Some(get) = get_commands.get(&id) {
            let (out, _) = run_command(get, stop);
            if !pending.lock().unwrap().contains_key(&id)
                && !pending_mutes.lock().unwrap().contains_key(&id)
            {
                results.lock().unwrap().insert(id, out);
            }
        }
    };
    let pop_any = |map: &Mutex<HashMap<usize, String>>| {
        let mut map = map.lock().unwrap();
        map.keys().next().copied().map(|id| {
            let cmd = map.remove(&id).unwrap();
            (id, cmd)
        })
    };
    while !stop.is_set() {
        // Mutes drain first: an unmute queued together with a drag's first
        // value must apply before the volume does.
        if let Some((id, command)) = pop_any(&pending_mutes) {
            let _ = run_command(&command, &stop);
            refresh(id, &stop);
            continue;
        }
        let next = {
            let mut map = pending.lock().unwrap();
            map.keys().next().copied().map(|id| {
                let value = map.remove(&id).unwrap();
                (id, value)
            })
        };
        match next {
            Some((id, value)) => {
                if let Some(command) = commands.get(&id) {
                    // Reuses the widget command runner for its timeout and
                    // stop handling; the output is irrelevant.
                    let _ = run_command(&fill_placeholder(command, &value.to_string()), &stop);
                }
                refresh(id, &stop);
            }
            // Queued work sets `wake`; a value queued while a command ran
            // left it set, so this returns at once.
            None => wake.take(SETTER_CHECK),
        }
    }
}

impl Drop for WidgetRuntime {
    fn drop(&mut self) {
        self.stop.set();
        self.setter_wake.set();
        for refresh in &self.refresh {
            refresh.set();
        }
    }
}

fn run_widget(
    spec: WidgetSpec,
    results: Arc<Mutex<HashMap<usize, WidgetOutput>>>,
    stop: Arc<Signal>,
    refresh: Arc<Signal>,
    wake: Arc<OwnedFd>,
) {
    let mut retry = RETRY_BASE;
    while !stop.is_set() {
        let (output, ok) = run_command(&spec.command, &stop);
        let changed = {
            let mut map = results.lock().unwrap();
            if map.get(&spec.id) != Some(&output) {
                map.insert(spec.id, output);
                true
            } else {
                false
            }
        };
        if changed {
            // Wake the main loop. One byte is enough; it drains all of them.
            let byte = [1u8];
            unsafe {
                libc::write(wake.as_raw_fd(), byte.as_ptr() as *const c_void, 1);
            }
        }
        let delay = if ok {
            retry = RETRY_BASE;
            spec.interval
        } else {
            let delay = retry.min(spec.interval);
            retry = (retry * 2).min(spec.interval.max(RETRY_BASE));
            delay
        };
        // Returns early on a tap refresh or when the runtime stops.
        refresh.take(delay);
    }
}

/// Run `sh -c <command>`, returning its parsed output and whether it exited
/// zero. Kills the child if it exceeds the timeout or the runtime is stopping.
fn run_command(command: &str, stop: &Signal) -> (WidgetOutput, bool) {
    let child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            // The output is discarded for slider set/mute commands, so the
            // log line is the only trace a spawn failure leaves (e.g. no `sh`
            // on the unit's PATH).
            eprintln!("not-quite-tiny-dfr: failed to run command {command:?}: {e}");
            return (
                WidgetOutput {
                    text: format!("err: {e}"),
                    color: None,
                    icon: None,
                },
                false,
            )
        }
    };
    let start = Instant::now();
    let mut ok = false;
    let mut out = Vec::new();
    // Block on the script's stdout rather than polling its exit: the output is
    // read as it arrives, and EOF means the script is done. A script that
    // blocks (e.g. battery.sh -w waiting on UPower) then costs no wakeups but
    // the periodic timeout/stop check.
    let mut stdout = child.stdout.take();
    let mut reap_wait = Duration::from_millis(1);
    loop {
        if stdout.is_none() {
            match child.try_wait() {
                Ok(Some(status)) => {
                    ok = status.success();
                    break;
                }
                Ok(None) => {}
                Err(_) => break,
            }
        }
        let timed_out = start.elapsed() > COMMAND_TIMEOUT;
        if timed_out || stop.is_set() {
            if timed_out {
                eprintln!(
                    "not-quite-tiny-dfr: command timed out after {}s: {command:?}",
                    COMMAND_TIMEOUT.as_secs()
                );
            }
            let _ = child.kill();
            let _ = child.wait();
            return if timed_out {
                (
                    WidgetOutput {
                        text: "timeout".into(),
                        color: None,
                        icon: None,
                    },
                    false,
                )
            } else {
                (WidgetOutput::default(), false)
            };
        }
        match &mut stdout {
            Some(pipe) => {
                let remaining = COMMAND_TIMEOUT.saturating_sub(start.elapsed());
                let mut pfd = libc::pollfd {
                    fd: pipe.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                };
                let ms = CHILD_CHECK.min(remaining).as_millis().max(1) as i32;
                if unsafe { libc::poll(&mut pfd, 1, ms) } > 0 {
                    let mut buf = [0u8; 4096];
                    match pipe.read(&mut buf) {
                        Ok(0) | Err(_) => stdout = None,
                        Ok(n) => out.extend_from_slice(&buf[..n]),
                    }
                }
            }
            // Stdout closed: the script is exiting; reap it with a short,
            // growing back-off.
            None => {
                stop.wait(reap_wait);
                reap_wait = (reap_wait * 2).min(CHILD_CHECK);
            }
        }
    }
    (parse_output(&String::from_utf8_lossy(&out)), ok)
}

/// Parse a widget's stdout: JSON `{"text","color"}` if it looks like JSON,
/// otherwise the first non-empty line as plain text.
fn parse_output(raw: &str) -> WidgetOutput {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return WidgetOutput::default();
    }
    if trimmed.starts_with('{') {
        if let Ok(j) = serde_json::from_str::<WidgetJson>(trimmed) {
            return WidgetOutput {
                text: j.text.unwrap_or_default(),
                color: j.color.as_deref().and_then(Color::parse_hex),
                icon: j.icon.filter(|s| !s.is_empty()),
            };
        }
        // Looked like JSON but wasn't valid; fall through to plain text.
    }
    WidgetOutput {
        text: trimmed.lines().next().unwrap_or_default().to_string(),
        color: None,
        icon: None,
    }
}

/// Set `O_NONBLOCK` on a fd (the wake pipe's read end, so draining never blocks).
pub fn set_nonblocking(fd: RawFd) {
    unsafe {
        let flags = libc::fcntl(fd, libc::F_GETFL);
        if flags >= 0 {
            libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
        }
    }
}

/// Drain any pending wake bytes from the pipe read end (non-blocking).
pub fn drain(fd: RawFd) {
    let mut buf = [0u8; 64];
    loop {
        let n = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut c_void, buf.len()) };
        if n <= 0 {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_text_first_line() {
        let o = parse_output("42%\nignored\n");
        assert_eq!(o.text, "42%");
        assert_eq!(o.color, None);
    }

    #[test]
    fn parses_json_text_and_color() {
        let o = parse_output(r##"{"text":"42%","color":"#ff0000"}"##);
        assert_eq!(o.text, "42%");
        assert_eq!(o.color, Some(Color::rgb(1.0, 0.0, 0.0)));
        assert_eq!(o.icon, None);
    }

    #[test]
    fn parses_json_icon() {
        let o = parse_output(r##"{"text":"69%","icon":"battery_5_bar"}"##);
        assert_eq!(o.text, "69%");
        assert_eq!(o.icon.as_deref(), Some("battery_5_bar"));
        // An empty icon string is treated as absent.
        assert_eq!(parse_output(r##"{"text":"x","icon":""}"##).icon, None);
        // Plain text carries no icon.
        assert_eq!(parse_output("50%").icon, None);
    }

    #[test]
    fn invalid_json_falls_back_to_text() {
        assert_eq!(parse_output("{not json").text, "{not json");
        assert_eq!(parse_output("   ").text, ""); // whitespace -> empty
    }

    #[test]
    fn interval_is_clamped_to_floor_and_defaulted() {
        assert_eq!(WidgetSpec::interval_from_secs(Some(0.001)), MIN_INTERVAL);
        assert_eq!(WidgetSpec::interval_from_secs(Some(-5.0)), MIN_INTERVAL);
        assert_eq!(WidgetSpec::interval_from_secs(None), Duration::from_secs(2));
        assert_eq!(
            WidgetSpec::interval_from_secs(Some(5.0)),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn run_command_returns_output_when_script_exits() {
        let stop = Signal::default();
        let t = Instant::now();
        let (o, ok) = run_command("sleep 0.3; echo done", &stop);
        assert!(ok);
        assert_eq!(o.text, "done");
        assert!(t.elapsed() < Duration::from_millis(900), "{:?}", t.elapsed());
    }

    #[test]
    fn run_command_reports_failure() {
        let stop = Signal::default();
        let (_, ok) = run_command("exit 3", &stop);
        assert!(!ok);
    }

    #[test]
    fn stop_interrupts_a_running_script() {
        let stop = Arc::new(Signal::default());
        let s = stop.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            s.set();
        });
        let t = Instant::now();
        let (_, ok) = run_command("sleep 20", &stop);
        assert!(!ok);
        assert!(t.elapsed() < Duration::from_millis(1500), "{:?}", t.elapsed());
    }

    #[test]
    fn signal_wait_returns_on_set() {
        let sig = Arc::new(Signal::default());
        let s = sig.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            s.set();
        });
        let t = Instant::now();
        assert!(sig.wait(Duration::from_secs(10)));
        assert!(t.elapsed() < Duration::from_secs(1));
    }
}
