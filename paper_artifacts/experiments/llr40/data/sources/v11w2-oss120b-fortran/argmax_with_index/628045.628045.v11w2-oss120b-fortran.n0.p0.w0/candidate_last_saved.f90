module argmax_mod
  use iso_c_binding
  implicit none
contains
  subroutine argmax_with_index_fp64(a, out_index, out_value, len_1d) bind(C, name="argmax_with_index_fp64")
    ! Arguments
    real(c_double), intent(in) :: a(*)
    integer(c_int64_t), intent(out) :: out_index(1)
    real(c_double), intent(out) :: out_value(1)
    integer(c_int64_t), value :: len_1d

    ! Local variables
    real(c_double) :: x, max_local
    integer(c_int64_t) :: i, idx_local

    if (len_1d <= 0) then
      out_index(1) = 0_c_int64_t   ! sentinel (no element)
      out_value(1) = 0.0_c_double
      return
    end if

    ! Initialise global result with first element (Fortran 1‑based index)
    out_value(1) = a(1)
    out_index(1) = 1_c_int64_t

    !$omp parallel private(i, x, max_local, idx_local)
      max_local = -huge(1.0_c_double)
      idx_local = 0_c_int64_t
      !$omp do schedule(static)
      do i = 1, len_1d
        x = a(i)
        if (x > max_local) then
          max_local = x
          idx_local = i   ! store Fortran 1‑based index
        end if
      end do
      !$omp end do

      !$omp critical
        if (max_local > out_value(1)) then
          out_value(1) = max_local
          out_index(1) = idx_local
        end if
      !$omp end critical
    !$omp end parallel

  end subroutine argmax_with_index_fp64
end module argmax_mod
