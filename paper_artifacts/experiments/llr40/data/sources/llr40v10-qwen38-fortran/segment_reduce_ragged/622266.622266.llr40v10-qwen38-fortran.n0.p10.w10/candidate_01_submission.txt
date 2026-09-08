! Segmented dot product over a ragged CSR-style structure.
! C-ABI entry: segment_reduce_ragged_fp64(double *out, int64_t *row_ptr, double *val,
!                                          double *w, int64_t nseg, uint8_t *ws, int64_t ws_size)
subroutine segment_reduce_ragged_fp64(out_c, rp_c, val_c, w_c, nseg, ws_c, ws_size) &
     bind(C, name="segment_reduce_ragged_fp64")
  use iso_c_binding
  implicit none
  type(c_ptr), value, intent(in) :: out_c, rp_c, val_c, w_c, ws_c
  integer(c_int64_t), value, intent(in) :: nseg, ws_size
  real(c_double), dimension(:), pointer :: val, w, out
  integer(c_int64_t), dimension(:), pointer :: rp
  integer :: nseg_i, s, e
  integer(c_int64_t) :: lo, hi
  real(c_double) :: acc

  if (nseg <= 0) return
  nseg_i = int(nseg)
  call c_f_pointer(rp_c, rp, [nseg_i + 1])
  call c_f_pointer(val_c, val, [int(rp(nseg_i + 1))])
  call c_f_pointer(w_c, w, [int(rp(nseg_i + 1))])
  call c_f_pointer(out_c, out, [nseg_i])

  !$omp parallel do schedule(dynamic) default(none) shared(rp, val, w, out, nseg_i) private(s, e, lo, hi, acc)
  do s = 1, nseg_i
     acc = 0.0d0
     lo = rp(s)
     hi = rp(s + 1)
     do e = int(lo), int(hi - 1)
        acc = acc + val(e + 1) * w(e + 1)
     end do
     out(s) = acc
  end do
end subroutine segment_reduce_ragged_fp64
