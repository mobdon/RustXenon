//! Networking, reported as "not connected": the stack starts, there is no
//! link, and every socket call fails with "network down".

use super::*;

const SOCKET_ERROR: u32 = u32::MAX;
const WSAENETDOWN: u32 = 10050;

fn fail(_: &Runtime, c: &mut Context) {
    ret(c, SOCKET_ERROR);
}

pub fn net_wsa_startup(rt: &Runtime, c: &mut Context) {
    let data = arg(rt, c, 2);
    if data != 0 {
        fill(rt, data, 0x190, 0);
        rt.mem.write_u16(data, 0x0202);
        rt.mem.write_u16(data + 2, 0x0202);
    }
    ret(c, 0);
}
pub fn net_wsa_cleanup(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn net_wsa_get_last_error(_: &Runtime, c: &mut Context) {
    ret(c, WSAENETDOWN);
}
pub fn net_x_net_startup(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn net_x_net_cleanup(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
/// No cable.
pub fn net_x_net_get_ethernet_link_status(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
/// XNET_GET_XNADDR_NONE
pub fn net_x_net_get_title_xn_addr(rt: &Runtime, c: &mut Context) {
    let p = arg(rt, c, 1);
    if p != 0 {
        fill(rt, p, 0x24, 0);
    }
    ret(c, 1);
}
/// XNET_CONNECT_STATUS_IDLE
pub fn net_x_net_get_connect_status(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn net_inet_addr(rt: &Runtime, c: &mut Context) {
    let s = cstr(rt, arg(rt, c, 0));
    let v = s.parse::<std::net::Ipv4Addr>().map_or(u32::MAX, u32::from);
    ret(c, v);
}
pub fn net_socket(_: &Runtime, c: &mut Context) {
    ret(c, u32::MAX); // INVALID_SOCKET
}
pub fn net_closesocket(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn net_shutdown(rt: &Runtime, c: &mut Context) {
    fail(rt, c)
}
pub fn net_ioctlsocket(rt: &Runtime, c: &mut Context) {
    fail(rt, c)
}
pub fn net_setsockopt(rt: &Runtime, c: &mut Context) {
    fail(rt, c)
}
pub fn net_getsockname(rt: &Runtime, c: &mut Context) {
    fail(rt, c)
}
pub fn net_bind(rt: &Runtime, c: &mut Context) {
    fail(rt, c)
}
pub fn net_connect(rt: &Runtime, c: &mut Context) {
    fail(rt, c)
}
pub fn net_listen(rt: &Runtime, c: &mut Context) {
    fail(rt, c)
}
pub fn net_accept(_: &Runtime, c: &mut Context) {
    ret(c, u32::MAX)
}
pub fn net_select(rt: &Runtime, c: &mut Context) {
    fail(rt, c)
}
pub fn net_recv(rt: &Runtime, c: &mut Context) {
    fail(rt, c)
}
pub fn net_recvfrom(rt: &Runtime, c: &mut Context) {
    fail(rt, c)
}
pub fn net_send(rt: &Runtime, c: &mut Context) {
    fail(rt, c)
}
pub fn net_sendto(rt: &Runtime, c: &mut Context) {
    fail(rt, c)
}
/// The secure-address and quality-of-service calls fail: WSAENETDOWN.
pub fn net_x_net_xn_addr_to_in_addr(_: &Runtime, c: &mut Context) {
    ret(c, WSAENETDOWN);
}
pub fn net_x_net_server_to_in_addr(_: &Runtime, c: &mut Context) {
    ret(c, WSAENETDOWN);
}
pub fn net_x_net_unregister_in_addr(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn net_x_net_qos_listen(_: &Runtime, c: &mut Context) {
    ret(c, WSAENETDOWN);
}
pub fn net_x_net_qos_lookup(_: &Runtime, c: &mut Context) {
    ret(c, WSAENETDOWN);
}
pub fn net_x_net_qos_release(_: &Runtime, c: &mut Context) {
    ret(c, 0);
}
pub fn x_net_logon_get_machine_id(_: &Runtime, c: &mut Context) {
    ret(c, 0x8015_1000);
}
pub fn x_net_logon_get_title_id(_: &Runtime, c: &mut Context) {
    ret(c, 0x8015_1000);
}
