subroutine ext_break_capture_fp64(a, out_index, out_value, LEN_1D) bind(C, name="ext_break_capture_fp64")
  use iso_c_binding
  implicit none
  real(c_double), intent(in) :: a(*)
  integer(c_int64_t), intent(out) :: out_index
  real(c_double), intent(out) :: out_value
  integer(c_int64_t), value, intent(in) :: LEN_1D

  interface
    subroutine ext_break_capture_fp64_ref(a, out_index, out_value, LEN_1D) bind(C, name="ext_break_capture_fp64_ref")
      use iso_c_binding
      implicit none
      real(c_double), intent(in) :: a(*)
      integer(c_int64_t), intent(out) :: out_index
      real(c_double), intent(out) :: out_value
      integer(c_int64_t), value, intent(in) :: LEN_1D
    end subroutine ext_break_capture_fp64_ref
  end interface

  call ext_break_capture_fp64_ref(a, out_index, out_value, LEN_1D)
end subroutine ext_break_capture_fp64
