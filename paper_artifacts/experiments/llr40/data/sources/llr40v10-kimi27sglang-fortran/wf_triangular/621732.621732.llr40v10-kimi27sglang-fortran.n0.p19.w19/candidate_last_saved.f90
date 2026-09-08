module wf_triangular_mod
  use iso_c_binding, only: c_int64_t, c_double, c_ptr, c_f_pointer
  implicit none
contains
  subroutine wf_triangular_fp64(a_c, LEN_2D) bind(C, name="wf_triangular_fp64")
    type(c_ptr), value, intent(in) :: a_c
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), pointer :: a(:,:)
    integer(c_int64_t) :: n, i0, j0
    integer(c_int64_t) :: nt, ti, tj
    integer(c_int64_t) :: i_start, i_end, j_start, j_end, js
    integer(c_int64_t), parameter :: TILE = 128

    n = LEN_2D
    if (n <= 1) return

    ! a(j+1, i+1) maps to C offset i*n + j in the row-major buffer.
    call c_f_pointer(a_c, a, [n, n])

    ! For small problems use the direct loop nest.
    if (n < 512) then
      do i0 = 1_c_int64_t, n - 1_c_int64_t
        do j0 = i0, n - 1_c_int64_t
          a(j0 + 1, i0 + 1) = a(j0 + 1, i0 + 1) + a(j0 + 1, i0) + a(j0, i0 + 1)
        end do
      end do
      return
    end if

    nt = (n + TILE - 1_c_int64_t) / TILE

    do ti = 0_c_int64_t, nt - 1_c_int64_t
      i_start = ti * TILE
      i_end   = min(i_start + TILE - 1_c_int64_t, n - 1_c_int64_t)
      if (i_start == 0_c_int64_t) i_start = 1_c_int64_t
      if (i_start > i_end) cycle

      do tj = ti, nt - 1_c_int64_t
        j_start = tj * TILE
        j_end   = min(j_start + TILE - 1_c_int64_t, n - 1_c_int64_t)

        do i0 = i_start, i_end
          js = max(i0, j_start)
          if (js > j_end) cycle
          do j0 = js, j_end
            a(j0 + 1, i0 + 1) = a(j0 + 1, i0 + 1) + a(j0 + 1, i0) + a(j0, i0 + 1)
          end do
        end do
      end do
    end do
  end subroutine wf_triangular_fp64
end module wf_triangular_mod
